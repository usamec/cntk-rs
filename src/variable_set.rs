use variable::Variable;
use shape::Shape;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>
  #include <unordered_set>

  using namespace CNTK;
  using namespace std;
}}

type VariableSetInner = [u64; 1usize];

/// Wrapper around unordered_set<Variable>
pub struct VariableSet {
    pub(super) payload: *mut VariableSetInner
}

impl VariableSet {
    /// Creates empty VariableSet
    pub fn new() -> VariableSet {
        VariableSet {
            payload: unsafe {
                cpp!([] -> *mut VariableSetInner as "unordered_set<Variable>*" {
                    return new unordered_set<Variable>;
                })
            }
        }
    }

    /// Adds Variable to set
    pub fn add<T: Into<Variable>>(&mut self, variable: T) {
        let vv = variable.into();
        let var_payload = vv.payload;
        let mut payload = self.payload;

        unsafe {
            cpp!([mut payload as "unordered_set<Variable>*", var_payload as "Variable"] {
                payload->insert(var_payload);
            })
        }
    }
}

impl Drop for VariableSet {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "unordered_set<Variable>*"] {
                delete payload;
            })
        };
    }
}

#[macro_export]
macro_rules! variableset {
    (@single $($x:tt)*) => (());
    (@count $($rest:expr),*) => (<[()]>::len(&[$(variableset!(@single $rest)),*]));
    
    ($($key:expr,)+) => { variableset!($($key),+) };
    ($($key:expr),*) => {
        {
            let mut _set = VariableSet::new();
            $(
                _set.add($key);
            )*
            _set
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use variable::*;

    #[test]
    fn test_create() {
        let set = VariableSet::new();
    }

    #[test]
    fn test_add() {
        let mut set = VariableSet::new();
        let var = Variable::input_variable(&Shape::scalar());

        set.add(&var);
    }
}