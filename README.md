# Bindings for CNTK library

Simple low level bindings for [CNTK library](https://github.com/Microsoft/CNTK/blob/release/2.2/Source/CNTKv2LibraryDll/API/CNTKLibrary.h) from Microsoft.

## Status

Currently exploring ways how to interact with C++ API nicely from Rust.
Expect a lot of breaking changes.

Build scripts are not ready yet, might not work outside of 64bit linux.

## Plans

* Figure out how to pass data in and out.
* Implement most of graph building functionality. Probably using some sort of code generation.
Goal is to have either recurrent or seq2seq network training possible.