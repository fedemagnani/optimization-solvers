use super::*;

pub type BoxedLayer<S> = Box<dyn Layer<S> + Send + Sync>;

#[derive(Default, Copy, Clone)]
/// Enum with different log formats, passed in the building process
pub enum LogFormat {
    /// Pretty format, very detailed (also with line number where log is emitted)
    Pretty,
    /// Json format
    Json,
    /// Normal format
    #[default]
    Normal,
}

#[derive(Default)]
pub struct Tracer {
    std_out_layer: Option<BoxedLayer<Registry>>,
    journald_layer: Option<BoxedLayer<Registry>>,
    file_layer: Option<BoxedLayer<Registry>>,
    _guards: Vec<WorkerGuard>,
}

impl Tracer {
    /// Append a layer for write logs to stdout with dedicated thread
    pub fn with_stdout_layer(mut self, format: Option<LogFormat>) -> Self {
        let (writer, guard) = tracing_appender::non_blocking(std::io::stdout());
        let format = format.unwrap_or_default();
        let std_out_layer: BoxedLayer<Registry> = match format {
            LogFormat::Pretty => Box::new(fmt::layer().pretty().with_writer(writer)),
            LogFormat::Json => Box::new(fmt::layer().json().with_writer(writer)),
            LogFormat::Normal => Box::new(fmt::layer().with_writer(writer)),
        };
        self.std_out_layer = Some(std_out_layer);
        self._guards.push(guard);
        self
    }

    /// Builds a new Tracer with the layers set in the building steps. Don't drop the guards!
    pub fn build(self) -> Vec<WorkerGuard> {
        let env_filter = EnvFilter::from_default_env();
        let mut layers = vec![];
        if let Some(std_out_layer) = self.std_out_layer {
            layers.push(std_out_layer);
        }
        if let Some(journald_layer) = self.journald_layer {
            layers.push(journald_layer);
        }
        if let Some(file_layer) = self.file_layer {
            layers.push(file_layer);
        }
        tracing_subscriber::registry()
            .with(layers)
            .with(env_filter)
            .init();
        self._guards
    }
}
