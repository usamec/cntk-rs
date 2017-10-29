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
    pub fn new() -> Variable {
        let payload = unsafe {
            cpp!([] -> VariableInner as "Variable" {
                auto inputVarName = L"features";
                const size_t inputDim = 5;
                return InputVariable({ inputDim }, DataType::Float, inputVarName);
            })
        };
        Variable { payload }
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