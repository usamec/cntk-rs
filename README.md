# Bindings for CNTK library

Simple low level bindings for [CNTK library](https://github.com/Microsoft/CNTK/blob/release/2.2/Source/CNTKv2LibraryDll/API/CNTKLibrary.h) from Microsoft.

## Status

Currently exploring ways how to interact with C++ API nicely from Rust.
Expect a lot of breaking changes.

Build scripts are not ready yet, might not work outside of 64bit linux.

## Other limitations

Only works with single precision (f32 in Rust, float in C++) types.
Only works with dense representations of vectors/matrices/tensors.

## Roadmap

* Figure out how to pass data in and out. Currently `Value::batch` for input and `Value::to_vec` should be enough.
* Variable creation.
* Gradients.
* Training simple feed forward net.
* Training recurrent or seq2seq net.
* Code for most operations. (probably generated).
* Write some meaningful examples.