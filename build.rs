extern crate cpp_build;

use cpp_build::Config;

fn main() {
    Config::new().flag("-std=c++11").build("src/lib.rs"); 
    println!("cargo:rustc-link-lib=dylib=Cntk.Core-2.4");
}
