use variable::Variable;
use value::{Value, ValueInner};
use shape::Shape;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>
  #include <unordered_map>

  using namespace CNTK;
  using namespace std;
}}

type DataMapInner = [u64; 1usize];

/// Wrapper around unordered_map<Variable, Value> to pass bindings to Function evaluation
pub struct DataMap {
    pub(super) payload: *mut DataMapInner
}

impl DataMap {
    /// Creates an empty DataMap
    pub fn new() -> DataMap {
        DataMap {
            payload: unsafe {
                cpp!([] -> *mut DataMapInner as "unordered_map<Variable, ValuePtr>*" {
                    return new unordered_map<Variable, ValuePtr>;
                })
            }
        }
    }

    /// Adds binding to DataMap. If mapping for given Variable exists, it will be overwritten.
    pub fn add(&mut self, variable: &Variable, value: &Value) {
        let var_payload = variable.payload;
        let val_payload = value.payload;
        let mut payload = self.payload;

        unsafe {
            cpp!([mut payload as "unordered_map<Variable, ValuePtr>*", var_payload as "Variable", val_payload as "ValuePtr"] {
                payload->insert({var_payload, val_payload});
            })
        }
    }

    /// Adds binding to null to DataMap. Useful, when we want function evaluation to create the Value.
    pub fn add_null(&mut self, variable: &Variable) {
        let var_payload = variable.payload;
        let mut payload = self.payload;

        unsafe {
            cpp!([mut payload as "unordered_map<Variable, ValuePtr>*", var_payload as "Variable"] {
                payload->insert({var_payload, nullptr});
            })
        }
    }

    pub fn get(&self, variable: &Variable) -> Option<Value> {
        let var_payload = variable.payload;
        let payload = self.payload;

        let has_var = unsafe {
            cpp!([payload as "unordered_map<Variable, ValuePtr>*", var_payload as "Variable"] -> bool as "bool" {
                return payload->count(var_payload);
            })
        };

        if has_var {
            Some(
                Value { payload: unsafe {
                    cpp!([payload as "unordered_map<Variable, ValuePtr>*", var_payload as "Variable"] -> ValueInner as "ValuePtr" {
                        return payload->find(var_payload)->second;
                    })
                }}
            )
        } else {
            None
        }
    }
}

impl Drop for DataMap {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "unordered_map<Variable, ValuePtr>*"] {
                delete payload;
            })
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use variable::*;
    use value::*;
    use device::*;

    #[test]
    fn test_create() {
        let map = DataMap::new();
    }

    #[test]
    fn test_add_and_get() {
        let mut map = DataMap::new();
        let var = Variable::input_variable(Shape::scalar());
        let var2 = Variable::input_variable(Shape::scalar());

        let data: Vec<f32> = vec!(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 110.0);

        let val = Value::batch(&var.shape(), &data, DeviceDescriptor::cpu());
        map.add(&var, &val);
        assert_eq!(map.get(&var).is_some(), true);
        assert_eq!(map.get(&var2).is_some(), false);
    }
}