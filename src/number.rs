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
        let mut x = self.clone();
        for i in 0..x.len() {
            if x[i] < lower_bound[i] {
                x[i] = lower_bound[i];
            } else if x[i] > upper_bound[i] {
                x[i] = upper_bound[i];
            }
        }
        x
    }
}
