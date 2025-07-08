use std::env;

fn main() {
    // Check if the wasm feature is enabled
    if env::var("CARGO_FEATURE_WASM").is_ok() {
        println!("cargo:warning=🚀 WASM feature enabled!");
        println!("cargo:warning=📦 To build WASM: cargo build --target wasm32-unknown-unknown");
        println!("cargo:warning=🌐 Or use: ./build-wasm.sh");
        println!("cargo:warning=📖 See wasm/README.md for usage instructions");
    }

    // Always rebuild if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}
