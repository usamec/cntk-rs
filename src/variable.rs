use shape::{Shape, ShapeInner};
use device::DeviceDescriptor;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

type ParameterInitializerInner = [u64; 2usize];

pub struct ParameterInitializer {
    payload: ParameterInitializerInner
}

impl ParameterInitializer {
    pub fn constant(value: f64) -> ParameterInitializer {
        ParameterInitializer {payload: unsafe {
            cpp!([value as "double"] -> ParameterInitializerInner as "ParameterInitializer" {
                return ConstantInitializer(value);
            })
        }}
    }

    pub fn uniform(scale: f64) -> ParameterInitializer {
        ParameterInitializer {payload: unsafe {
            cpp!([scale as "double"] -> ParameterInitializerInner as "ParameterInitializer" {
                return UniformInitializer(scale);
            })
        }}
    }

    pub fn normal(scale: f64) -> ParameterInitializer {
        ParameterInitializer {payload: unsafe {
            cpp!([scale as "double"] -> ParameterInitializerInner as "ParameterInitializer" {
                return NormalInitializer(scale);
            })
        }}
    }

    pub fn xavier() -> ParameterInitializer {
        ParameterInitializer {payload: unsafe {
            cpp!([] -> ParameterInitializerInner as "ParameterInitializer" {
                return XavierInitializer();
            })
        }}
    }

    pub fn glorot_uniform() -> ParameterInitializer {
        ParameterInitializer {payload: unsafe {
            cpp!([] -> ParameterInitializerInner as "ParameterInitializer" {
                return GlorotUniformInitializer();
            })
        }}
    }

    pub fn glorot_normal() -> ParameterInitializer {
        ParameterInitializer {payload: unsafe {
            cpp!([] -> ParameterInitializerInner as "ParameterInitializer" {
                return GlorotNormalInitializer();
            })
        }}
    }

    pub fn he_uniform() -> ParameterInitializer {
        ParameterInitializer {payload: unsafe {
            cpp!([] -> ParameterInitializerInner as "ParameterInitializer" {
                return HeUniformInitializer();
            })
        }}
    }

    pub fn he_normal() -> ParameterInitializer {
        ParameterInitializer {payload: unsafe {
            cpp!([] -> ParameterInitializerInner as "ParameterInitializer" {
                return HeNormalInitializer();
            })
        }}
    }
}

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

    pub fn parameter(shape: Shape, initializer: ParameterInitializer, device: DeviceDescriptor) -> Variable {
        let spayload = shape.payload;
        let dpayload = device.payload;
        let initializerpayload = initializer.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape", dpayload as "DeviceDescriptor", initializerpayload as "ParameterInitializer"] -> VariableInner as "Variable" {
                return Parameter(spayload, DataType::Float, initializerpayload, dpayload);
            })
        }}
    }

    pub fn placeholder(shape: Shape) -> Variable {
        let spayload = shape.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape"] -> VariableInner as "Variable" {
                return PlaceholderVariable(spayload);
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

    pub fn is_parameter(&self) -> bool {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "Variable"] -> bool as "bool" {
                return payload.IsParameter();
            })
        }
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