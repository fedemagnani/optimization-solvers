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
