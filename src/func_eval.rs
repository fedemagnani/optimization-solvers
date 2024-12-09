use super::*;

// Function evaluation structure. Builder pattern
#[derive(derive_getters::Getters, Debug)]
pub struct FuncEval<T, H> {
    f: Floating,
    g: T,
    hessian: Option<H>,
}

impl<T, H> FuncEval<T, H> {
    pub fn new(f: Floating, g: T) -> Self {
        FuncEval {
            f,
            g,
            hessian: None,
        }
    }
}

pub type FuncEvalUnivariate = FuncEval<Floating, Floating>;
pub type FuncEvalMultivariate = FuncEval<DVector<Floating>, DMatrix<Floating>>;

impl FuncEvalMultivariate {
    pub fn with_hessian(mut self, hessian: DMatrix<Floating>) -> Self {
        self.hessian = Some(hessian);
        self
    }
    pub fn take_hessian(&mut self) -> DMatrix<Floating> {
        self.hessian.take().unwrap()
    }
}

impl From<(Floating, DVector<Floating>)> for FuncEvalMultivariate {
    fn from(value: (Floating, DVector<Floating>)) -> Self {
        let (f, g) = value;
        FuncEvalMultivariate::new(f, g)
    }
}
