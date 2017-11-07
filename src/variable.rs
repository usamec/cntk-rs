use shape::{Shape, ShapeInner};
use device::DeviceDescriptor;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

pub(super) type VariableInner = [u64; 5usize];

#[derive(Debug)]
pub struct Variable {
    pub(super) payload: VariableInner
}

impl Variable {
    pub fn input_variable(shape: Shape) -> Variable {
        let spayload = shape.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape"] -> VariableInner as "Variable" {
                return InputVariable(spayload, DataType::Float);
            })
        }}
    }

    pub fn input_variable_with_name(shape: Shape, name: &str) -> Variable {
        let spayload = shape.payload;
        let name_ptr = name.as_ptr();
        let name_len = name.len();
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape", name_ptr as "char*", name_len as "size_t"] -> VariableInner as "Variable" {
                string name(name_ptr, name_ptr + name_len);
                wstring wname;
                wname.assign(name.begin(), name.end());
                return InputVariable(spayload, DataType::Float, wname);
            })
        }}
    }

    pub fn input_variable_with_gradient(shape: Shape) -> Variable {
        let spayload = shape.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape"] -> VariableInner as "Variable" {
                return InputVariable(spayload, DataType::Float, true);
            })
        }}
    }

    pub fn parameter(shape: Shape, device: DeviceDescriptor) -> Variable {
        let spayload = shape.payload;
        let dpayload = device.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape", dpayload as "DeviceDescriptor"] -> VariableInner as "Variable" {
                return Parameter(spayload, DataType::Float, GlorotUniformInitializer(), dpayload);
            })
        }}
    }

    pub fn shape(&self) -> Shape {
        let payload = self.payload;
        Shape { payload: unsafe {
            cpp!([payload as "Variable"] -> ShapeInner as "NDShape" {
                return payload.Shape();
            })
        }}
    }

    pub fn name(&self) -> String {
        let payload = self.payload;
        let name_size = unsafe {
            cpp!([payload as "Variable"] -> usize as "size_t" {
                auto wname = payload.Name();
                string name(wname.begin(), wname.end());
                return name.size();
            })
        };
        let mut bytes = Vec::with_capacity(name_size);
        unsafe {
            bytes.set_len(name_size);
            let mut ptr = bytes.as_mut_ptr();
            cpp!([payload as "Variable", mut ptr as "char*"] {
                auto wname = payload.Name();
                string name(wname.begin(), wname.end());
                copy(name.begin(), name.end(), ptr);
            })
        }
        String::from_utf8(bytes).unwrap()
    }
}

pub fn plus(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return Plus(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn elem_times(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return ElementTimes(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn times(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return Times(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn cross_entropy_with_softmax(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return CrossEntropyWithSoftmax(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn squared_error(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return SquaredError(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn classification_error(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return ClassificationError(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn tanh(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Tanh(xpayload);
        })
    };
    Variable {payload}
}

pub fn reduce_sum_all(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return ReduceSum(xpayload, Axis::AllAxes());
        })
    };
    Variable {payload}
}

pub fn named_alias(x: &Variable, name: &str) -> Variable {
    let xpayload = x.payload;
    let name_ptr = name.as_ptr();
    let name_len = name.len();
    Variable { payload: unsafe {
        cpp!([xpayload as "Variable", name_ptr as "char*", name_len as "size_t"] -> VariableInner as "Variable" {
                string name(name_ptr, name_ptr + name_len);
                wstring wname;
                wname.assign(name.begin(), name.end());
                return Alias(xpayload, wname);
            })
    }}
}


impl Clone for Variable {
    fn clone(&self) -> Self {
        let xpayload = self.payload;
        let payload = unsafe {
            cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
                return xpayload;
            })
        };
        Variable {payload}
    }
}

impl Drop for Variable {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "Variable"] {
                payload.~Variable();
            })
        };
    }
}