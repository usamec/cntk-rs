use variable::{Variable, VariableInner};
use variable_set::VariableSet;
use data_map::DataMap;
use device::DeviceDescriptor;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

type BackPropStateInner = [u64; 2usize];

type FunctionInner = [u64; 2usize];

#[derive(Debug)]
pub struct Function {
    pub(super) payload: FunctionInner
}

pub struct BackPropState {
    payload: BackPropStateInner
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
        if self.num_outputs() > 1 {
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

    pub fn forward(&self, input_data_map: &DataMap, output_data_map: &mut DataMap, device: DeviceDescriptor,
                   retain_backward_state_for: &VariableSet, exclude_gradients_for: &VariableSet) -> BackPropState {
        let payload = self.payload;
        let impayload = input_data_map.payload;
        let mut ompayload = output_data_map.payload;
        let dpayload = device.payload;
        let bspayload = retain_backward_state_for.payload;
        let egpayload = exclude_gradients_for.payload;
        BackPropState { payload: unsafe {
            cpp!([payload as "FunctionPtr", impayload as "unordered_map<Variable, ValuePtr>*", mut ompayload as "unordered_map<Variable, ValuePtr>*", dpayload as "DeviceDescriptor",
                  bspayload as "unordered_set<Variable>*", egpayload as "unordered_set<Variable>*"] -> BackPropStateInner as "BackPropStatePtr" {
                return payload->Forward(*impayload, *ompayload, dpayload, *bspayload, *egpayload);
            })
        }}
    }

    pub fn backward(&self, bpstate: &BackPropState, gradient_values: &DataMap, output_map: &mut DataMap) {
        let payload = self.payload;
        let bppayload = bpstate.payload;
        let gpayload = gradient_values.payload;
        let mut opayload = output_map.payload;
        // TODO: check if requested variables allow gradients
        unsafe {
            cpp!([payload as "FunctionPtr", bppayload as "BackPropStatePtr", gpayload as "unordered_map<Variable, ValuePtr>*", mut opayload as "unordered_map<Variable, ValuePtr>*"] {
                payload->Backward(bppayload, *gpayload, *opayload);
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

impl Drop for BackPropState {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "BackPropStatePtr"] {
                payload.~BackPropStatePtr();
            })
        };
    }
}