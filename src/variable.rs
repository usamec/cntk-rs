use shape::Shape;

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