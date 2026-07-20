use std::cell::RefCell;
use std::collections::HashMap;

use num_traits::Zero;
use pjrt::ProgramFormat::MLIR;
use pjrt::{Buffer, Client, HostBuffer, LoadedExecutable, Program};
use serde::{de::DeserializeOwned, Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    error::DiffsolError, ConstantOp, ConstantOpSensAdjoint, Context, MatrixHost, NonLinearOp,
    NonLinearOpAdjoint, NonLinearOpJacobian, NonLinearOpSensAdjoint, OdeEquations, OdeEquationsRef,
    Op, Scalar, UnitCallable, Vector, VectorHost,
};

pub const PLUGIN_ENV: &str = "DIFFSOL_PJRT_PLUGIN";

pub trait StableHloScalar: Scalar + Serialize + DeserializeOwned {
    fn host_buffer(data: Vec<Self>, dims: Option<Vec<i64>>) -> HostBuffer;
    fn buffer_as_slice(buf: &HostBuffer) -> Option<&[Self]>;
}

macro_rules! impl_stablehlo_scalar {
    ($t:ty, $variant:ident) => {
        impl StableHloScalar for $t {
            fn host_buffer(data: Vec<Self>, dims: Option<Vec<i64>>) -> HostBuffer {
                HostBuffer::builder().data(data).maybe_dims(dims).build()
            }
            fn buffer_as_slice(buf: &HostBuffer) -> Option<&[Self]> {
                match buf {
                    HostBuffer::$variant(b) => Some(b.data()),
                    _ => None,
                }
            }
        }
    };
}

impl_stablehlo_scalar!(f32, F32);
impl_stablehlo_scalar!(f64, F64);

fn load_client(plugin_path: Option<&str>) -> Result<Client, DiffsolError> {
    let path = match plugin_path {
        Some(p) => p.to_string(),
        None => std::env::var(PLUGIN_ENV).map_err(|_| {
            DiffsolError::Other(format!(
                "no PJRT plugin path provided and ${PLUGIN_ENV} is not set"
            ))
        })?,
    };
    let api = pjrt::plugin(&path)
        .load()
        .map_err(|e| DiffsolError::Other(format!("failed to load PJRT plugin `{path}`: {e}")))?;
    Client::builder(&api)
        .build()
        .map_err(|e| DiffsolError::Other(format!("failed to create PJRT client: {e}")))
}

struct Executable {
    name: String,
    exec: LoadedExecutable,
    kept_var_idx: Vec<usize>,
}

enum Arg<'a, T> {
    Scalar(T),
    Vec(&'a [T]),
}

impl Executable {
    fn compile(
        client: &Client,
        name: &str,
        stablehlo: &[u8],
        kept_var_idx: Vec<usize>,
    ) -> Result<Self, DiffsolError> {
        let program = Program::new(MLIR, stablehlo);
        let exec = LoadedExecutable::builder(client, &program)
            .build()
            .map_err(|e| DiffsolError::Other(format!("PJRT compile `{name}`: {e}")))?;
        Ok(Self {
            name: name.to_string(),
            exec,
            kept_var_idx,
        })
    }

    fn run<T: StableHloScalar>(
        &self,
        client: &Client,
        args: &[Arg<'_, T>],
        out: &mut [T],
    ) -> Result<(), DiffsolError> {
        let err = |msg: String| DiffsolError::Other(format!("module `{}`: {msg}", self.name));
        let mut inputs: Vec<Buffer> = Vec::with_capacity(self.kept_var_idx.len());
        for &i in &self.kept_var_idx {
            let arg = args
                .get(i)
                .ok_or_else(|| err(format!("kept_var_idx {i} out of range")))?;
            let host = match arg {
                Arg::Scalar(x) => T::host_buffer(vec![*x], Some(Vec::new())),
                Arg::Vec(xs) => T::host_buffer(xs.to_vec(), None),
            };
            inputs.push(
                host.copy_to_sync(client)
                    .map_err(|e| err(format!("host-to-device copy failed: {e}")))?,
            );
        }
        let result = self
            .exec
            .execution(inputs)
            .run_sync()
            .map_err(|e| err(format!("execute failed: {e}")))?;
        let output = result[0][0]
            .copy_to_host_sync()
            .map_err(|e| err(format!("device-to-host copy failed: {e}")))?;
        let data = T::buffer_as_slice(&output)
            .ok_or_else(|| err("unexpected output element type".into()))?;
        if data.len() != out.len() {
            return Err(err(format!(
                "output length {} does not match expected {}",
                data.len(),
                out.len()
            )));
        }
        out.copy_from_slice(data);
        Ok(())
    }
}

/// ODE equations whose right-hand side and derivative operators are StableHLO
/// modules executed through PJRT.
///
/// The expected modules (keyed by name in [`StableHloEquations::from_parts`]) are:
///
/// | name | logical signature | diffsol trait method |
/// |---|---|---|
/// | `rhs` | `(t, u, p) -> f` | [`NonLinearOp::call_inplace`] |
/// | `jac_mul` | `(t, u, p, v) -> (df/du) v` | [`NonLinearOpJacobian::jac_mul_inplace`] |
/// | `jac_transpose_mul` | `(t, u, p, v) -> -(df/du)^T v` | [`NonLinearOpAdjoint::jac_transpose_mul_inplace`] |
/// | `sens_transpose_mul` | `(t, u, p, v) -> -(df/dp)^T v` | [`NonLinearOpSensAdjoint::sens_transpose_mul_inplace`] |
/// | `jacobian` (optional) | `(t, u, p) -> df/du` dense `(n, n)` row-major | [`NonLinearOpJacobian::jacobian_inplace`] |
///
/// The initial condition `u0` is a constant captured at construction and is
/// assumed *not* to depend on the parameters `p` (as is the case for a JAX
/// caller that computes `u0` itself); accordingly the initial-condition
/// parameter sensitivities are identically zero.
pub struct StableHloEquations<M: MatrixHost<T: StableHloScalar>> {
    client: Client,
    rhs: Executable,
    jac_mul: Executable,
    jac_transpose_mul: Executable,
    sens_transpose_mul: Executable,
    jacobian: Option<Executable>,
    modules: HashMap<String, (Vec<u8>, Vec<usize>)>,
    p: RefCell<Vec<M::T>>,
    y0: M::V,
    nparams: usize,
}

impl<M: MatrixHost<T: StableHloScalar>> StableHloEquations<M> {
    pub fn from_parts(
        modules: &HashMap<String, (Vec<u8>, Vec<usize>)>,
        u0: &[M::T],
        p: &[M::T],
        plugin_path: Option<&str>,
        ctx: M::C,
    ) -> Result<Self, DiffsolError> {
        let client = load_client(plugin_path)?;

        let compile = |name: &str| -> Result<Executable, DiffsolError> {
            let (bytes, kept) = modules
                .get(name)
                .ok_or_else(|| DiffsolError::Other(format!("module `{name}` missing")))?;
            Executable::compile(&client, name, bytes, kept.clone())
        };

        let y0 = ctx.vector_from_vec(u0.to_vec());

        Ok(Self {
            rhs: compile("rhs")?,
            jac_mul: compile("jac_mul")?,
            jac_transpose_mul: compile("jac_transpose_mul")?,
            sens_transpose_mul: compile("sens_transpose_mul")?,
            jacobian: modules
                .contains_key("jacobian")
                .then(|| compile("jacobian"))
                .transpose()?,
            client,
            modules: modules.clone(),
            p: RefCell::new(p.to_vec()),
            y0,
            nparams: p.len(),
        })
    }

    fn run(&self, exec: &Executable, t: M::T, u: &M::V, v: Option<&M::V>, out: &mut M::V) {
        let p = self.p.borrow();
        let result = match v {
            Some(v) => exec.run(
                &self.client,
                &[
                    Arg::Scalar(t),
                    Arg::Vec(u.as_slice()),
                    Arg::Vec(&p),
                    Arg::Vec(v.as_slice()),
                ],
                out.as_mut_slice(),
            ),
            None => exec.run(
                &self.client,
                &[Arg::Scalar(t), Arg::Vec(u.as_slice()), Arg::Vec(&p)],
                out.as_mut_slice(),
            ),
        };
        result.unwrap_or_else(|e| panic!("PJRT execute `{}`: {e}", exec.name));
    }
}

impl<M: MatrixHost<T: StableHloScalar>> Serialize for StableHloEquations<M> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Payload<'a, T> {
            modules: &'a HashMap<String, (Vec<u8>, Vec<usize>)>,
            u0: &'a [T],
            p: &'a [T],
        }
        let p = self.p.borrow();
        Payload {
            modules: &self.modules,
            u0: self.y0.as_slice(),
            p: &p,
        }
        .serialize(serializer)
    }
}

impl<'de, M: MatrixHost<T: StableHloScalar>> Deserialize<'de> for StableHloEquations<M> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Payload<T> {
            modules: HashMap<String, (Vec<u8>, Vec<usize>)>,
            u0: Vec<T>,
            p: Vec<T>,
        }
        let payload = Payload::<M::T>::deserialize(deserializer)?;
        Self::from_parts(
            &payload.modules,
            &payload.u0,
            &payload.p,
            None,
            M::C::default(),
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<M: MatrixHost<T: StableHloScalar>> Op for StableHloEquations<M> {
    type M = M;
    type T = M::T;
    type V = M::V;
    type C = M::C;

    fn nstates(&self) -> usize {
        self.y0.len()
    }
    fn nout(&self) -> usize {
        self.y0.len()
    }
    fn nparams(&self) -> usize {
        self.nparams
    }
    fn context(&self) -> &Self::C {
        self.y0.context()
    }
}

impl<'a, M: MatrixHost<T: StableHloScalar>> OdeEquationsRef<'a> for StableHloEquations<M> {
    type Mass = UnitCallable<M>;
    type Rhs = StableHloRhs<'a, M>;
    type Root = UnitCallable<M>;
    type Init = StableHloInit<'a, M>;
    type Out = UnitCallable<M>;
    type Reset = UnitCallable<M>;
}

impl<M: MatrixHost<T: StableHloScalar>> OdeEquations for StableHloEquations<M> {
    fn rhs(&self) -> StableHloRhs<'_, M> {
        StableHloRhs(self)
    }

    fn mass(&self) -> Option<UnitCallable<M>> {
        None
    }

    fn init(&self) -> StableHloInit<'_, M> {
        StableHloInit(self)
    }

    fn set_params(&mut self, p: &Self::V) {
        self.p
            .borrow_mut()
            .iter_mut()
            .zip(p.as_slice().iter())
            .for_each(|(dst, src)| *dst = *src);
    }

    fn get_params(&self, p: &mut Self::V) {
        p.as_mut_slice()
            .iter_mut()
            .zip(self.p.borrow().iter())
            .for_each(|(dst, src)| *dst = *src);
    }
}

pub struct StableHloRhs<'a, M: MatrixHost<T: StableHloScalar>>(&'a StableHloEquations<M>);
pub struct StableHloInit<'a, M: MatrixHost<T: StableHloScalar>>(&'a StableHloEquations<M>);

macro_rules! impl_op_for_stablehlo {
    ($name:ident) => {
        impl<M: MatrixHost<T: StableHloScalar>> Op for $name<'_, M> {
            type M = M;
            type T = M::T;
            type V = M::V;
            type C = M::C;

            fn nstates(&self) -> usize {
                self.0.nstates()
            }
            fn nout(&self) -> usize {
                self.0.nout()
            }
            fn nparams(&self) -> usize {
                self.0.nparams()
            }
            fn context(&self) -> &Self::C {
                self.0.context()
            }
        }
    };
}

impl_op_for_stablehlo!(StableHloRhs);
impl_op_for_stablehlo!(StableHloInit);

impl<M: MatrixHost<T: StableHloScalar>> NonLinearOp for StableHloRhs<'_, M> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.0.run(&self.0.rhs, t, x, None, y);
    }
}

impl<M: MatrixHost<T: StableHloScalar>> NonLinearOpJacobian for StableHloRhs<'_, M> {
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.0.run(&self.0.jac_mul, t, x, Some(v), y);
    }

    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(jacobian) = &self.0.jacobian {
            let n = self.nstates();
            let mut flat = vec![M::T::zero(); n * n];
            {
                let p = self.0.p.borrow();
                jacobian
                    .run(
                        &self.0.client,
                        &[Arg::Scalar(t), Arg::Vec(x.as_slice()), Arg::Vec(&p)],
                        &mut flat,
                    )
                    .unwrap_or_else(|e| panic!("PJRT execute `jacobian`: {e}"));
            }
            let mut col = M::V::zeros(n, self.context().clone());
            for j in 0..n {
                for (i, c) in col.as_mut_slice().iter_mut().enumerate() {
                    *c = flat[i * n + j];
                }
                y.set_column(j, &col);
            }
        } else {
            self._default_jacobian_inplace(x, t, y);
        }
    }
}

impl<M: MatrixHost<T: StableHloScalar>> NonLinearOpAdjoint for StableHloRhs<'_, M> {
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.0.run(&self.0.jac_transpose_mul, t, x, Some(v), y);
    }
}

impl<M: MatrixHost<T: StableHloScalar>> NonLinearOpSensAdjoint for StableHloRhs<'_, M> {
    fn sens_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.0.run(&self.0.sens_transpose_mul, t, x, Some(v), y);
    }
}

impl<M: MatrixHost<T: StableHloScalar>> ConstantOp for StableHloInit<'_, M> {
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.copy_from(&self.0.y0);
    }
}

impl<M: MatrixHost<T: StableHloScalar>> ConstantOpSensAdjoint for StableHloInit<'_, M> {
    fn sens_transpose_mul_inplace(&self, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(M::T::zero());
    }
}
