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