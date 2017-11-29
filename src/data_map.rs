use variable::Variable;
use value::{Value, ValueInner};

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
    pub fn add<T: Into<Variable>>(&mut self, variable: T, value: &Value) {
        let v = variable.into();
        let var_payload = v.payload;
        let val_payload = value.payload;
        let mut payload = self.payload;

        unsafe {
            cpp!([mut payload as "unordered_map<Variable, ValuePtr>*", var_payload as "Variable", val_payload as "ValuePtr"] {
                payload->insert({var_payload, val_payload});
            })
        }
    }

    /// Adds binding to null to DataMap. Useful, when we want function evaluation to create the Value.
    pub fn add_null<T: Into<Variable>>(&mut self, variable: T) {
        let v = variable.into();
        let var_payload = v.payload;
        let mut payload = self.payload;

        unsafe {
            cpp!([mut payload as "unordered_map<Variable, ValuePtr>*", var_payload as "Variable"] {
                payload->insert({var_payload, nullptr});
            })
        }
    }

    pub fn get<T: Into<Variable>>(&self, variable: T) -> Option<Value> {
        let v = variable.into();
        let var_payload = v.payload;
        let payload = self.payload;

        let has_var = unsafe {
            cpp!([payload as "unordered_map<Variable, ValuePtr>*", var_payload as "Variable"] -> bool as "bool" {
                auto it = payload->find(var_payload);
                if (it == payload->end()) return false;
                if (it->second.get() == NULL) return false;
                return true;
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

#[macro_export]
macro_rules! datamap {
    (@single $($x:tt)*) => (());
    (@count $($rest:expr),*) => (<[()]>::len(&[$(datamap!(@single $rest)),*]));

    ($($key:expr => $value:expr,)+) => { datamap!($($key => $value),+) };
    ($($key:expr => $value:expr),*) => {
        {
            let mut _map = DataMap::new();
            $(
                _map.add($key, $value);
            )*
            _map
        }
    };
}

#[macro_export]
macro_rules! outdatamap {
    (@single $($x:tt)*) => (());
    (@count $($rest:expr),*) => (<[()]>::len(&[$(outdatamap!(@single $rest)),*]));
    
    ($($key:expr,)+) => { outdatamap!($($key),+) };
    ($($key:expr),*) => {
        {
            let mut _set = DataMap::new();
            $(
                _set.add_null($key);
            )*
            _set
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use variable::*;
    use value::*;
    use device::*;
    use shape::Shape;

    #[test]
    fn test_create() {
        let _map = DataMap::new();
    }

    #[test]
    fn test_add_and_get() {
        let mut map = DataMap::new();
        let var = Variable::input_variable(&Shape::scalar());
        let var2 = Variable::input_variable(&Shape::scalar());

        let data: Vec<f32> = vec!(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 110.0);

        let val = Value::batch_from_vec(&var.shape(), &data, DeviceDescriptor::cpu());
        map.add(var.clone(), &val);
        assert_eq!(map.get(var).is_some(), true);
        assert_eq!(map.get(var2).is_some(), false);
    }
}