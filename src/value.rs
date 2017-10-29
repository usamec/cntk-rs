use variable::Variable;
use device::DeviceDescriptor;
use std::ptr;

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
    pub fn batch(var: &Variable, data: &[f32], device: DeviceDescriptor) -> Value {
        let data_ptr = data.as_ptr();
        let data_size = data.len();
        let var_payload = var.payload;
        let device_payload = device.payload;
        let payload = unsafe {
            cpp!([var_payload as "Variable", data_ptr as "float*", data_size as "size_t", device_payload as "DeviceDescriptor" ] -> ValueInner as "ValuePtr" {
            vector<float> data(data_ptr, data_ptr + data_size);
            return Value::CreateBatch(var_payload.Shape(), data, device_payload);
        })
        };
        Value { payload }
    }

    pub fn empty() -> Value {
        Value { payload: unsafe {
            cpp!([] -> ValueInner as "ValuePtr" {
                ValuePtr ptr;
                return ptr;
            })
        }}
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