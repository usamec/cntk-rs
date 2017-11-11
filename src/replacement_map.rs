use variable::Variable;
use shape::Shape;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>
  #include <unordered_map>

  using namespace CNTK;
  using namespace std;
}}

type ReplacementMapInner = [u64; 1usize];

/// Wrapper around unordered_map<Variable, Variable> to pass replacement for placeholders
pub struct ReplacementMap {
    pub(super) payload: *mut ReplacementMapInner
}

impl ReplacementMap {
    /// Creates an empty ReplacementMap
    pub fn new() -> ReplacementMap {
        ReplacementMap {
            payload: unsafe {
                cpp!([] -> *mut ReplacementMapInner as "unordered_map<Variable, Variable>*" {
                    return new unordered_map<Variable, Variable>;
                })
            }
        }
    }

    /// Adds mapping to ReplacementMap. If mapping for given Variable exists, it will be overwritten.
    pub fn add<T: Into<Variable>>(&mut self, variable: &Variable, replacement: T) {
        let var_payload = variable.payload;
        let rv = replacement.into();
        let repl_payload = rv.payload;
        let mut payload = self.payload;

        unsafe {
            cpp!([mut payload as "unordered_map<Variable, Variable>*", var_payload as "Variable", repl_payload as "Variable"] {
                payload->insert({var_payload, repl_payload});
            })
        }
    }
}

impl Drop for ReplacementMap {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "unordered_map<Variable, Variable>*"] {
                delete payload;
            })
        };
    }
}

#[macro_export]
macro_rules! replacementmap {
    (@single $($x:tt)*) => (());
    (@count $($rest:expr),*) => (<[()]>::len(&[$(replacementmap!(@single $rest)),*]));

    ($($key:expr => $value:expr,)+) => { replacementmap!($($key => $value),+) };
    ($($key:expr => $value:expr),*) => {
        {
            let mut _map = ReplacementMap::new();
            $(
                _map.add($key, $value);
            )*
            _map
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use variable::*;
    use value::*;
    use device::*;

    #[test]
    fn test_create() {
        let map = ReplacementMap::new();
    }

    #[test]
    fn test_add_and_get() {
        let mut map = ReplacementMap::new();
        let var = Variable::input_variable(&Shape::scalar());
        let var2 = Variable::input_variable(&Shape::scalar());
        
        map.add(&var, &var2);
    }
}