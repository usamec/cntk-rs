use variable::{Variable, VariableInner};
use variable_set::VariableSet;
use data_map::DataMap;
use replacement_map::ReplacementMap;
use device::DeviceDescriptor;
use std::ptr;
use std::ffi::CStr;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

type BackPropStateInner = [u64; 2usize];

pub(super) type FunctionInner = [u64; 2usize];

#[derive(Debug)]
pub struct Function {
    pub(super) payload: FunctionInner
}

pub struct BackPropState {
    payload: BackPropStateInner
}

impl Function {
    pub fn from_variable<T: Into<Variable>>(var: T) -> Function {
        let varv = var.into();
        let payload = varv.payload;
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

    pub fn to_variable(&self) -> Result<Variable, &'static str> {
        let payload = self.payload;
        if self.num_outputs() > 1 {
            Err("Cannot convert function with multiple outputs into Variable")
        } else {
            Ok(Variable { payload: unsafe {
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
            let mut error_p: *mut i8 = ptr::null_mut();
            cpp!([payload as "FunctionPtr", impayload as "unordered_map<Variable, ValuePtr>*", mut ompayload as "unordered_map<Variable, ValuePtr>*", dpayload as "DeviceDescriptor", mut error_p as "char*"] {
                try {
                    payload->Evaluate(*impayload, *ompayload, dpayload);
                } catch (std::exception& e) {
                    auto what = e.what();
                    error_p = new char[strlen(what)+1];
                    strcpy(error_p, what);
                }
            });
            if !error_p.is_null() {
                let msg = CStr::from_ptr(error_p).to_str().unwrap();
                panic!("{}", msg);
            }
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
            let mut error_p: *mut i8 = ptr::null_mut();
            let payload = cpp!([payload as "FunctionPtr", impayload as "unordered_map<Variable, ValuePtr>*", mut ompayload as "unordered_map<Variable, ValuePtr>*", dpayload as "DeviceDescriptor",
                  bspayload as "unordered_set<Variable>*", egpayload as "unordered_set<Variable>*", mut error_p as "char*"] -> BackPropStateInner as "BackPropStatePtr" {
                try {
                    return payload->Forward(*impayload, *ompayload, dpayload, *bspayload, *egpayload);
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
        }}
    }

    pub fn backward(&self, bpstate: &BackPropState, gradient_values: &DataMap, output_map: &mut DataMap) {
        let payload = self.payload;
        let bppayload = bpstate.payload;
        let gpayload = gradient_values.payload;
        let mut opayload = output_map.payload;
        // TODO: check if requested variables allow gradients
        unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            cpp!([payload as "FunctionPtr", bppayload as "BackPropStatePtr", gpayload as "unordered_map<Variable, ValuePtr>*", mut opayload as "unordered_map<Variable, ValuePtr>*", mut error_p as "char*"] {
                try {
                    payload->Backward(bppayload, *gpayload, *opayload);
                } catch (std::exception& e) {
                    auto what = e.what();
                    error_p = new char[strlen(what)+1];
                    strcpy(error_p, what);
                }
            });
            if !error_p.is_null() {
                let msg = CStr::from_ptr(error_p).to_str().unwrap();
                panic!("{}", msg);
            }
        };
    }

    pub fn save(&self, path: &str) {
        let path_ptr = path.as_ptr();
        let path_len = path.len();
        let payload = self.payload;
        unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            cpp!([payload as "FunctionPtr", path_ptr as "char*", path_len as "size_t", mut error_p as "char*"] {
                try {
                    string path(path_ptr, path_ptr + path_len);
                    wstring wpath;
                    wpath.assign(path.begin(), path.end());
                    payload->Save(wpath);
                } catch (std::exception& e) {
                    auto what = e.what();
                    error_p = new char[strlen(what)+1];
                    strcpy(error_p, what);
                }
            });
            if !error_p.is_null() {
                let msg = CStr::from_ptr(error_p).to_str().unwrap();
                panic!("{}", msg);
            }
        }
    }

    pub fn load(path: &str, device: DeviceDescriptor) -> Function {
        let path_ptr = path.as_ptr();
        let path_len = path.len();
        let dpayload = device.payload;
        Function {payload: unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            let payload = cpp!([path_ptr as "char*", path_len as "size_t", dpayload as "DeviceDescriptor", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
                try {
                    string path(path_ptr, path_ptr + path_len);
                    wstring wpath;
                    wpath.assign(path.begin(), path.end());
                    return Function::Load(wpath, dpayload);
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
        let mut output: Vec<VariableInner> = Vec::with_capacity(num_inputs);
        unsafe {
            output.set_len(num_inputs);
            let mut ptr = output.as_mut_ptr();
            cpp!([payload as "FunctionPtr", mut ptr as "Variable*"] {
                auto outputs = payload->Inputs();
                for (size_t i = 0; i < outputs.size(); i++) {
                    ::new (&ptr[i]) Variable(outputs[i]);
                }
            })
        }
        output.into_iter().map(|x| Variable {payload: x}).collect::<Vec<Variable>>()
    }

    pub fn outputs(&self) -> Vec<Variable> {
        let payload = self.payload;
        let num_outputs = self.num_outputs();
        let mut output: Vec<VariableInner> = Vec::with_capacity(num_outputs);
        unsafe {
            output.set_len(num_outputs);
            let mut ptr = output.as_mut_ptr();
            cpp!([payload as "FunctionPtr", mut ptr as "Variable*"] {
                auto outputs = payload->Outputs();
                for (size_t i = 0; i < outputs.size(); i++) {
                    ::new (&ptr[i]) Variable(outputs[i]);
                }
            })
        }
        output.into_iter().map(|x| Variable {payload: x}).collect::<Vec<Variable>>()
    }

    pub fn replace_placeholders(self, placeholder_replacements: &ReplacementMap) -> Function {
        let payload = self.payload;
        let repl_payload = placeholder_replacements.payload;
        Function {payload: unsafe {
            let mut error_p: *mut i8 = ptr::null_mut();
            let payload = cpp!([payload as "FunctionPtr", repl_payload as "unordered_map<Variable, Variable>*", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
                try {
                    return payload->ReplacePlaceholders(*repl_payload);
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
        let mut output: Vec<VariableInner> = Vec::with_capacity(num_parameters);
        unsafe {
            output.set_len(num_parameters);
            let mut ptr = output.as_mut_ptr();
            cpp!([payload as "FunctionPtr", mut ptr as "Variable*"] {
                auto outputs = payload->Parameters();
                for (size_t i = 0; i < outputs.size(); i++) {
                    ::new (&ptr[i]) Variable(outputs[i]);
                }
            })
        }
        output.into_iter().map(|x| Variable {payload: x}).collect::<Vec<Variable>>()
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