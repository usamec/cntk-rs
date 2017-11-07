# Bindings for CNTK library

Simple low level bindings for [CNTK library](https://github.com/Microsoft/CNTK/blob/release/2.2/Source/CNTKv2LibraryDll/API/CNTKLibrary.h) from Microsoft.

## Status

Currently exploring ways how to interact with C++ API nicely from Rust.
Expect a lot of breaking changes.

Build scripts are not ready yet, might not work outside of 64bit linux.

## Building and installing

You need to have CNTK-2.2 installed and paths to includes and library files in
relevant enviroment variables (cntk activate scripts does this well).
You also need g++-4.8 installed (because CNTK uses it to compile things).

## Other limitations

Only works with single precision (f32 in Rust, float in C++) types.
Only works with dense representations of vectors/matrices/tensors.
Only works with ASCII strings for variable names and filenames.

## Roadmap

* Figure out how to pass data in and out. Currently `Value::batch` for input and `Value::to_vec` should be enough.
* Variable creation. - Some basics are there
* Gradients. - Possible to do backward pass
* Training simple feed forward net. - Possible.
* Save and load model. - Possible.
* Training recurrent or seq2seq net. - Recurrences are possible.
* Convnets.
* Code for most operations. (probably generated). - Currently only selected ones are supported.
* Write some meaningful examples. - Only in tests now.
* Catch all relevant C++ exceptions
* Better build scripts.