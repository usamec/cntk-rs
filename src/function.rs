use variable::{Variable, VariableInner};
use variable_set::VariableSet;
use data_map::DataMap;
use replacement_map::ReplacementMap;
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

    pub fn combine(variables: &[&Variable]) -> Function {
        let data: Vec<Variable> = variables.iter().map(|&x| x.clone()).collect();
        let data_ptr = data.as_ptr();
        let data_size = data.len();
        Function { payload: unsafe {
            cpp!([data_ptr as "Variable*", data_size as "size_t"] -> FunctionInner as "FunctionPtr" {
                return Combine(vector<Variable>(data_ptr, data_ptr + data_size));
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

    pub fn save(&self, path: &str) {
        let path_ptr = path.as_ptr();
        let path_len = path.len();
        let payload = self.payload;
        unsafe {
            cpp!([payload as "FunctionPtr", path_ptr as "char*", path_len as "size_t"] {
                string path(path_ptr, path_ptr + path_len);
                wstring wpath;
                wpath.assign(path.begin(), path.end());
                payload->Save(wpath);
            })
        }
    }

    pub fn load(path: &str, device: DeviceDescriptor) -> Function {
        let path_ptr = path.as_ptr();
        let path_len = path.len();
        let dpayload = device.payload;
        Function {payload: unsafe {
            cpp!([path_ptr as "char*", path_len as "size_t", dpayload as "DeviceDescriptor"] -> FunctionInner as "FunctionPtr" {
                string path(path_ptr, path_ptr + path_len);
                wstring wpath;
                wpath.assign(path.begin(), path.end());
                return Function::Load(wpath, dpayload);
            })
        }}
    }

    pub fn num_inputs(&self) -> usize {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "FunctionPtr"] -> usize as "size_t" {
                return payload->Inputs().size();
            })
        }
    }

    pub fn inputs(&self) -> Vec<Variable> {
        let payload = self.payload;
        let num_inputs = self.num_inputs();
        let mut output = Vec::with_capacity(num_inputs);
        unsafe {
            output.set_len(num_inputs);
            let mut ptr = output.as_mut_ptr();
            cpp!([payload as "FunctionPtr", mut ptr as "Variable*"] {
                auto outputs = payload->Inputs();
                copy(outputs.begin(), outputs.end(), ptr);
            })
        }
        output
    }

    pub fn outputs(&self) -> Vec<Variable> {
        let payload = self.payload;
        let num_outputs = self.num_outputs();
        let mut output = Vec::with_capacity(num_outputs);
        unsafe {
            output.set_len(num_outputs);
            let mut ptr = output.as_mut_ptr();
            cpp!([payload as "FunctionPtr", mut ptr as "Variable*"] {
                auto outputs = payload->Outputs();
                copy(outputs.begin(), outputs.end(), ptr);
            })
        }
        output
    }

    pub fn replace_placeholders(self, placeholder_replacements: &ReplacementMap) -> Function {
        let payload = self.payload;
        let repl_payload = placeholder_replacements.payload;
        Function {payload: unsafe {
            cpp!([payload as "FunctionPtr", repl_payload as "unordered_map<Variable, Variable>*"] -> FunctionInner as "FunctionPtr" {
                return payload->ReplacePlaceholders(*repl_payload);
            })
        }}
    }

    pub fn num_parameters(&self) -> usize {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "FunctionPtr"] -> usize as "size_t" {
                return payload->Parameters().size();
            })
        }
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let payload = self.payload;
        let num_parameters = self.num_parameters();
        let mut output = Vec::with_capacity(num_parameters);
        unsafe {
            output.set_len(num_parameters);
            let mut ptr = output.as_mut_ptr();
            cpp!([payload as "FunctionPtr", mut ptr as "Variable*"] {
                auto outputs = payload->Parameters();
                copy(outputs.begin(), outputs.end(), ptr);
            })
        }
        output
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