use super::*;

// Function evaluation structure. Builder pattern
#[derive(derive_getters::Getters, Debug)]
pub struct FuncEval {
    f: Floating,
    g: DVector<Floating>,
    hessian: Option<DMatrix<Floating>>,
}
impl FuncEval {
    pub fn new(f: Floating, g: DVector<Floating>) -> Self {
        FuncEval {
            f,
            g,
            hessian: None,
        }
    }
    pub fn with_hessian(mut self, hessian: DMatrix<Floating>) -> Self {
        self.hessian = Some(hessian);
        self
    }
}

impl From<(Floating, DVector<Floating>)> for FuncEval {
    fn from(value: (Floating, DVector<Floating>)) -> Self {
        let (f, g) = value;
        FuncEval::new(f, g)
    }
}
