use variable::Variable;
use device::DeviceDescriptor;

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