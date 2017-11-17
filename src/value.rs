use shape::{Shape, ShapeInner};
use device::DeviceDescriptor;
use std::ptr;
use std::borrow::Borrow;
use std::ffi::CStr;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

pub(super) type ValueInner = [u64; 2usize];

#[derive(Debug)]
pub struct Value {
    pub(super) payload: ValueInner
}

impl Value {
    pub fn batch(shape: &Shape, data: &[f32], device: DeviceDescriptor) -> Value {
        let data_ptr = data.as_ptr();
        let data_size = data.len();
        let shape_payload = shape.payload;
        let device_payload = device.payload;
        let payload = unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            let payload = cpp!([shape_payload as "NDShape", data_ptr as "float*", data_size as "size_t", device_payload as "DeviceDescriptor", mut error_p as "char*" ] -> ValueInner as "ValuePtr" {
                try {
                    vector<float> data(data_ptr, data_ptr + data_size);
                    return Value::CreateBatch(shape_payload, data, device_payload);
                } catch (std::exception& e) {
                    auto what = e.what();
                    error_p = new char[strlen(what)+1];
                    strcpy(error_p, what);
                    return nullptr;
                }
            });
            if !error_p.is_null() {
                let msg = CStr::from_ptr(error_p).to_str().unwrap();
                panic!("{}", msg);
            }
            payload
        };
        Value { payload }
    }

    pub fn sequence(shape: &Shape, data: &[f32], device: DeviceDescriptor) -> Value {
        let data_ptr = data.as_ptr();
        let data_size = data.len();
        let shape_payload = shape.payload;
        let device_payload = device.payload;
        let payload = unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            let payload = cpp!([shape_payload as "NDShape", data_ptr as "float*", data_size as "size_t", device_payload as "DeviceDescriptor", mut error_p as "char*" ] -> ValueInner as "ValuePtr" {
                try {
                    vector<float> data(data_ptr, data_ptr + data_size);
                    return Value::CreateSequence(shape_payload, data, device_payload);
                } catch (std::exception& e) {
                    auto what = e.what();
                    error_p = new char[strlen(what)+1];
                    strcpy(error_p, what);
                    return nullptr;
                }
            });
            if !error_p.is_null() {
                let msg = CStr::from_ptr(error_p).to_str().unwrap();
                panic!("{}", msg);
            }
            payload
        };
        Value { payload }
    }

    pub fn from_vec(shape: &Shape, data: &[f32], device: DeviceDescriptor) -> Value {
        let data_ptr = data.as_ptr();
        let data_size = data.len();
        let shape_payload = shape.payload;
        let device_payload = device.payload;
        let payload = unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            let payload = cpp!([shape_payload as "NDShape", data_ptr as "float*", data_size as "size_t", device_payload as "DeviceDescriptor", mut error_p as "char*" ] -> ValueInner as "ValuePtr" {
                try {
                    return MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(shape_payload, data_ptr, data_size, device_payload, true)->DeepClone());
                } catch (std::exception& e) {
                    auto what = e.what();
                    error_p = new char[strlen(what)+1];
                    strcpy(error_p, what);
                    return nullptr;
                }
            });
            if !error_p.is_null() {
                let msg = CStr::from_ptr(error_p).to_str().unwrap();
                panic!("{}", msg);
            }
            payload
        };
        Value { payload }
    }

    pub fn one_hot_seq(shape: &Shape, seq: &[usize], device: DeviceDescriptor) -> Value {
        let data_ptr = seq.as_ptr();
        let data_size = seq.len();
        let shape_payload = shape.payload;
        let device_payload = device.payload;
        let payload = unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            let payload = cpp!([shape_payload as "NDShape", data_ptr as "size_t*", data_size as "size_t", device_payload as "DeviceDescriptor", mut error_p as "char*" ] -> ValueInner as "ValuePtr" {
                try {
                    vector<size_t> data(data_ptr, data_ptr + data_size);
                    return Value::Create<float>(shape_payload, { data }, device_payload);
                } catch (std::exception& e) {
                    auto what = e.what();
                    error_p = new char[strlen(what)+1];
                    strcpy(error_p, what);
                    return nullptr;
                }
            });
            if !error_p.is_null() {
                let msg = CStr::from_ptr(error_p).to_str().unwrap();
                panic!("{}", msg);
            }
            payload
        };
        Value { payload }
    }

    pub fn batch_of_sequences<T: Borrow<[f32]>>(shape: &Shape, seqs: &[T], device: DeviceDescriptor) -> Value {
        let sizes = seqs.iter().map(|x| x.borrow().len()).collect::<Vec<usize>>();
        let sizes_ptr = sizes.as_ptr();
        let seqs_ptr = seqs.iter().map(|x| x.borrow().as_ptr()).collect::<Vec<_>>();
        let data_ptr = seqs_ptr.as_ptr();
        let n_batches = seqs.len();
        let shape_payload = shape.payload;
        let device_payload = device.payload;
        let payload = unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            let payload = cpp!([shape_payload as "NDShape", sizes_ptr as "size_t*", n_batches as "size_t", data_ptr as "float**", device_payload as "DeviceDescriptor", mut error_p as "char*" ] -> ValueInner as "ValuePtr" {
                try {
                    vector<vector<float>> data;
                    for (size_t i = 0; i < n_batches; i++) {
                        data.push_back(vector<float>(data_ptr[i], data_ptr[i] + sizes_ptr[i]));
                    }
                    return Value::CreateBatchOfSequences(shape_payload, data, device_payload, true);
                } catch (std::exception& e) {
                    auto what = e.what();
                    error_p = new char[strlen(what)+1];
                    strcpy(error_p, what);
                    return nullptr;
                }
            });
            if !error_p.is_null() {
                let msg = CStr::from_ptr(error_p).to_str().unwrap();
                panic!("{}", msg);
            }
            payload
        };
        Value { payload }
    }

    pub fn batch_of_one_hot_sequences<T: Borrow<[usize]>>(shape: &Shape, seqs: &[T], device: DeviceDescriptor) -> Value {
        let sizes = seqs.iter().map(|x| x.borrow().len()).collect::<Vec<usize>>();
        let sizes_ptr = sizes.as_ptr();
        let seqs_ptr = seqs.iter().map(|x| x.borrow().as_ptr()).collect::<Vec<_>>();
        let data_ptr = seqs_ptr.as_ptr();
        let n_batches = seqs.len();
        let shape_payload = shape.payload;
        let device_payload = device.payload;
        let payload = unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            let payload = cpp!([shape_payload as "NDShape", sizes_ptr as "size_t*", n_batches as "size_t", data_ptr as "size_t**", device_payload as "DeviceDescriptor", mut error_p as "char*"] -> ValueInner as "ValuePtr" {
                try {
                    vector<vector<size_t>> data;
                    for (size_t i = 0; i < n_batches; i++) {
                        data.push_back(vector<size_t>(data_ptr[i], data_ptr[i] + sizes_ptr[i]));
                    }
                    return Value::Create<float>(shape_payload, data, device_payload, true);
                } catch (std::exception& e) {
                    auto what = e.what();
                    error_p = new char[strlen(what)+1];
                    strcpy(error_p, what);
                    return nullptr;
                }
            });
            if !error_p.is_null() {
                let msg = CStr::from_ptr(error_p).to_str().unwrap();
                panic!("{}", msg);
            }
            payload
        };
        Value { payload }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let payload = self.payload;
        let total_size = unsafe {
            cpp!([payload as "ValuePtr"] -> usize as "size_t" {
                return payload->Data()->Shape().TotalSize();
            })
        };
        let mut buffer: Vec<f32> = Vec::with_capacity(total_size);
        unsafe { buffer.set_len(total_size); }
        let data = unsafe {
            cpp!([payload as "ValuePtr"] -> *const f32 as "const float*" {
              return payload->Data()->DataBuffer<float>();
            })
        };
        unsafe {
            ptr::copy(data, buffer.as_mut_ptr(), total_size);
        }
        buffer
    }

    pub fn shape(&self) -> Shape {
        let payload = self.payload;
        Shape { payload: unsafe {
            cpp!([payload as "ValuePtr"] -> ShapeInner as "NDShape" {
                return payload->Shape();
            })
        }}
    }
}

impl Drop for Value {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "ValuePtr"] {
                payload.~ValuePtr();
            })
        };
    }
}