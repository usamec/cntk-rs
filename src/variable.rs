use axis::Axis;
use shape::{Shape, ShapeInner};
use device::DeviceDescriptor;
use function::Function;
use std::borrow::Borrow;
use std::ptr;

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
    pub fn create<A: Borrow<Axis>>(shape: &Shape, is_sparse: bool, needs_gradient: bool, name: &str, dynamic_axes: &[A]) -> Variable {
        let spayload = shape.payload;
        let name_ptr = name.as_ptr();
        let name_len = name.len();
        let axis_payloads = dynamic_axes.iter().map(|x| x.borrow().payload).collect::<Vec<_>>();
        let axis_ptr = axis_payloads.as_ptr();
        let axis_len = axis_payloads.len();
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape", name_ptr as "char*", name_len as "size_t", is_sparse as "bool", needs_gradient as "bool", axis_ptr as "Axis*", axis_len as "size_t"] -> VariableInner as "Variable" {
                string name(name_ptr, name_ptr + name_len);
                wstring wname;
                wname.assign(name.begin(), name.end());
                return InputVariable(spayload, is_sparse, DataType::Float, needs_gradient, wname, vector<Axis>(axis_ptr, axis_ptr + axis_len));
            })
        }}
    }

    pub fn input_variable(shape: &Shape) -> Variable {
        let spayload = shape.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape"] -> VariableInner as "Variable" {
                return InputVariable(spayload, DataType::Float);
            })
        }}
    }

    pub fn sparse_input_variable(shape: &Shape) -> Variable {
        let spayload = shape.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape"] -> VariableInner as "Variable" {
                return InputVariable(spayload, true, DataType::Float);
            })
        }}
    }

    pub fn input_variable_with_name(shape: &Shape, name: &str) -> Variable {
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

    pub fn input_variable_with_gradient(shape: &Shape) -> Variable {
        let spayload = shape.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape"] -> VariableInner as "Variable" {
                return InputVariable(spayload, DataType::Float, true);
            })
        }}
    }

    pub fn parameter(shape: &Shape, initializer: &ParameterInitializer, device: DeviceDescriptor) -> Variable {
        let spayload = shape.payload;
        let dpayload = device.payload;
        let initializerpayload = initializer.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape", dpayload as "DeviceDescriptor", initializerpayload as "ParameterInitializer"] -> VariableInner as "Variable" {
                return Parameter(spayload, DataType::Float, initializerpayload, dpayload);
            })
        }}
    }

    pub fn placeholder(shape: &Shape) -> Variable {
        let spayload = shape.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape"] -> VariableInner as "Variable" {
                return PlaceholderVariable(spayload);
            })
        }}
    }

    pub fn constant_scalar(value: f32) -> Variable {
        Variable { payload: unsafe {
            cpp!([value as "float"] -> VariableInner as "Variable" {
                return Constant::Scalar(DataType::Float, value);
            })
        }}
    }

    pub fn constant_repeat(shape: &Shape, value: f32) -> Variable {
        let spayload = shape.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape", value as "float"] -> VariableInner as "Variable" {
                return Constant(spayload, DataType::Float, value);
            })
        }}
    }

    pub fn constant_from_slice(shape: &Shape, value: &[f32], device: DeviceDescriptor) -> Variable {
        let spayload = shape.payload;
        let value_ptr = value.as_ptr();
        let value_len = value.len();
        let dpayload = device.payload;
        Variable { payload: unsafe {
            cpp!([spayload as "NDShape", value_ptr as "float*", value_len as "size_t", dpayload as "DeviceDescriptor"] -> VariableInner as "Variable" {
                return Constant(MakeSharedObject<NDArrayView>(spayload, value_ptr, value_len, dpayload, true)->DeepClone());
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

    pub fn normal_random(x: &Shape, mean: f64, scale: f64) -> Variable {
        let xpayload = x.payload;
        let payload = unsafe {
            cpp!([xpayload as "NDShape", mean as "double", scale as "double"] -> VariableInner as "Variable" {
            return NormalRandom(xpayload, DataType::Float, mean, scale);
        })
        };
        Variable {payload}
    }

    pub fn bernoulli_random(x: &Shape, mean: f64) -> Variable {
        let xpayload = x.payload;
        let payload = unsafe {
            cpp!([xpayload as "NDShape", mean as "double"] -> VariableInner as "Variable" {
            return BernoulliRandom(xpayload, DataType::Float, mean);
        })
        };
        Variable {payload}
    }

    pub fn uniform_random(x: &Shape, low: f64, high: f64) -> Variable {
        let xpayload = x.payload;
        let payload = unsafe {
            cpp!([xpayload as "NDShape", low as "double", high as "double"] -> VariableInner as "Variable" {
            return UniformRandom(xpayload, DataType::Float, low, high);
        })
        };
        Variable {payload}
    }

    pub fn gumbel_random(x: &Shape, loc: f64, scale: f64) -> Variable {
        let xpayload = x.payload;
        let payload = unsafe {
            cpp!([xpayload as "NDShape", loc as "double", scale as "double"] -> VariableInner as "Variable" {
            return GumbelRandom(xpayload, DataType::Float, loc, scale);
        })
        };
        Variable {payload}
    }

    pub fn parameter_to_vec(&self) -> Vec<f32> {
        assert!(self.is_parameter());
        let payload = self.payload;
        let total_size = unsafe {
            cpp!([payload as "Parameter"] -> usize as "size_t" {
                return payload.Value()->Shape().TotalSize();
            })
        };
        let mut buffer: Vec<f32> = Vec::with_capacity(total_size);
        unsafe { buffer.set_len(total_size); }
        let data = unsafe {
            cpp!([payload as "Parameter"] -> *const f32 as "const float*" {
              return payload.Value()->DataBuffer<float>();
            })
        };
        unsafe {
            ptr::copy(data, buffer.as_mut_ptr(), total_size);
        }
        buffer
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

impl Drop for ParameterInitializer {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "ParameterInitializer"] {
                payload.~ParameterInitializer();
            })
        };
    }
}

impl<T: Borrow<Function>> From<T> for Variable {
    fn from(f: T) -> Variable {
        f.borrow().to_variable().unwrap()
    }
}

impl<'a> From<&'a Variable> for Variable {
    fn from(f: &'a Variable) -> Variable {
        f.clone()
    }
}

