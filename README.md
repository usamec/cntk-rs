# Bindings for CNTK library

Simple low level bindings for [CNTK library](https://github.com/Microsoft/CNTK/blob/release/2.2/Source/CNTKv2LibraryDll/API/CNTKLibrary.h) from Microsoft.

## Status

Currently exploring ways how to interact with C++ API nicely from Rust.
Expect a lot of breaking changes.

Build scripts are not ready yet, might not work outside of 64bit linux.

Also library is probably not an idiomatic Rust.

## Building and installing

You need to have CNTK-2.2 installed and paths to includes and library files in
relevant enviroment variables (cntk activate scripts does this well).
You also need g++-4.8 installed (because CNTK uses it to compile things).

## Other limitations

Only works with single precision (f32 in Rust, float in C++) types.
Only works with dense representations of vectors/matrices/tensors.
Only works with ASCII strings for variable names and filenames.

## What works

* Passing data in and out of computation.
* Backpropagation.
* Training fully connected feedforward, convolutional and recurrent network (have to test bidirectional though).
* Saving and loading the model.
* Code for most operations. - Almost all, except couple of helpers.

## Planned in future

* Constant variable.
* Demo of seq2seq model training.
* Finish all operations.
* Add more examples and documentation. (like seq2seq with word embeddings).
* Catch all relevant C++ exceptions
* Better build scripts.
* Figure out whether we want NDArrayView or go directly from Rust data to Value and back.
* Interop with some NDArray library.

