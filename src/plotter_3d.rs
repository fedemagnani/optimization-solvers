use super::*;
use plotly::common::Mode;
use plotly::layout::Layout;
use plotly::{Plot, Surface};

pub struct Plotter3d {
    mesh_size: usize,
    mesh_x: Vec<f64>,
    mesh_y: Vec<f64>,
    plot: Plot,
}
impl Plotter3d {
    pub fn new(min_value: f64, max_value: f64, mesh_size: usize) -> Self {
        let (mesh_x, mesh_y) = (0..mesh_size)
            .map(|i| {
                (
                    min_value + (max_value - min_value) * (i as f64) / (mesh_size as f64),
                    min_value + (max_value - min_value) * (i as f64) / (mesh_size as f64),
                )
            })
            .unzip();
        let mut plot = Plot::new();
        plot.set_layout(Layout::new().width(1600).height(1000));
        Plotter3d {
            mesh_size,
            mesh_x,
            mesh_y,
            plot,
        }
    }
    pub fn with_mesh_x(mut self, mesh_x: Vec<f64>) -> Self {
        self.mesh_x = mesh_x;
        self
    }
    pub fn with_mesh_y(mut self, mesh_y: Vec<f64>) -> Self {
        self.mesh_y = mesh_y;
        self
    }
    pub fn append_plot(
        mut self,
        oracle: &impl Fn(&DVector<f64>) -> FuncEvalMultivariate,
        title: &str,
        opacity: f64,
    ) -> Self {
        let (n, (x, y)) = (self.mesh_size, (self.mesh_x.clone(), self.mesh_y.clone()));

        let mut z = vec![vec![0.0; n]; n];

        for (i, x) in x.iter().enumerate() {
            for (j, y) in y.iter().enumerate() {
                let input = DVector::from_vec(vec![*x, *y]);
                z[i][j] = *oracle(&input).f();
            }
        }
        let surface = Surface::new(z)
            .x(x)
            .y(y)
            .name(title)
            .show_scale(true)
            .opacity(opacity);
        self.plot.add_trace(surface);
        self
    }
    pub fn append_scatter_points(
        mut self,
        // f: &impl Fn(f64, f64) -> f64,
        oracle: &impl Fn(&DVector<f64>) -> FuncEvalMultivariate,
        points: &[DVector<f64>],
        title: &str,
    ) -> Self {
        let n = points.len();
        let mut z = vec![0.0; n];
        let labels = (0..n)
            .map(|i| format!("Point {}", i))
            .collect::<Vec<String>>();
        for (i, input) in points.iter().enumerate() {
            // let input = DVector::from_vec(vec![*x, *y]);
            z[i] = *oracle(input).f();
        }
        let (x, y) = points.iter().cloned().map(|v| (v[0], v[1])).unzip();
        let scatter = plotly::Scatter3D::new(x, y, z)
            .mode(Mode::Markers)
            .name(title)
            .text_array(labels);
        self.plot.add_trace(scatter);
        self
    }
    pub fn set_title(mut self, title: &str) -> Self {
        let layout = self.plot.layout().clone().title(title).show_legend(true);

        self.plot.set_layout(layout);
        self
    }
    pub fn set_layout_size(mut self, width: usize, height: usize) -> Self {
        let layout = self.plot.layout().clone().width(width).height(height);
        self.plot.set_layout(layout);
        self
    }
    pub fn build(&self, filename: &str) {
        self.plot.write_html(filename);
    }
}
