use super::*;

pub type Floating = f64;

pub trait BoxProjection {
    fn box_projection(
        &self,
        lower_bound: &DVector<Floating>,
        upper_bound: &DVector<Floating>,
    ) -> DVector<Floating>;
}

impl BoxProjection for DVector<Floating> {
    fn box_projection(
        &self,
        lower_bound: &DVector<Floating>,
        upper_bound: &DVector<Floating>,
    ) -> DVector<Floating> {
        self.sup(lower_bound).inf(upper_bound)
    }
}

pub trait InfinityNorm {
    fn infinity_norm(&self) -> Floating;
}

impl InfinityNorm for DVector<Floating> {
    fn infinity_norm(&self) -> Floating {
        self.iter().fold(0.0f64, |acc, x| acc.max(x.abs()))
    }
}
