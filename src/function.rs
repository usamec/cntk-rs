use variable::{Variable, VariableInner};
use data_map::DataMap;
use device::DeviceDescriptor;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

type FunctionInner = [u64; 2usize];

#[derive(Debug)]
pub struct Function {
    pub(super) payload: FunctionInner
}

impl Function {
    pub fn from_variable(var: &Variable) -> Function {
        let payload = var.payload;
        Function { payload: unsafe {
            cpp!([payload as "Variable"] -> FunctionInner as "FunctionPtr" {
                return payload;
            })
        }}
    }

    pub fn num_outputs(&self) -> usize {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "FunctionPtr"] -> usize as "size_t" {
                return payload->Outputs().size();
            })
        }
    }

    pub fn to_variable(&self) -> Option<Variable> {
        let payload = self.payload;
        if (self.num_outputs() > 1) {
            None
        } else {
            Some(Variable { payload: unsafe {
                cpp!([payload as "FunctionPtr"] -> VariableInner as "Variable" {
                    return Variable(payload);
                })
            }})
        }
    }

    pub fn evaluate(&self, input_data_map: &DataMap, output_data_map: &mut DataMap, device: DeviceDescriptor) {
        let payload = self.payload;
        let impayload = input_data_map.payload;
        let mut ompayload = output_data_map.payload;
        let dpayload = device.payload;
        unsafe {
            cpp!([payload as "FunctionPtr", impayload as "unordered_map<Variable, ValuePtr>*", mut ompayload as "unordered_map<Variable, ValuePtr>*", dpayload as "DeviceDescriptor"] {
                payload->Evaluate(*impayload, *ompayload, dpayload);
            })
        };
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "FunctionPtr"] {
                payload.~FunctionPtr();
            })
        };
    }
}