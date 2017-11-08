cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

pub(super) type AxisInner = [u64; 3usize];

pub struct Axis {
    pub(super) payload: AxisInner
}

impl Axis {
    pub fn all() -> Axis {
        Axis {
            payload: unsafe {
                cpp!([] -> AxisInner as "Axis" {
                    return Axis::AllAxes();
                })
            }
        }
    }

    pub fn new(number: i32) -> Axis {
        Axis {
            payload: unsafe {
                cpp!([number as "int"] -> AxisInner as "Axis" {
                    return Axis(number);
                })
            }
        }
    }
}

impl Drop for Axis {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "Axis"] {
                payload.~Axis();
            })
        };
    }
}

impl Clone for Axis {
    fn clone(&self) -> Self {
        let xpayload = self.payload;
        let payload = unsafe {
            cpp!([xpayload as "Axis"] -> AxisInner as "Axis" {
                return xpayload;
            })
        };
        Axis {payload}
    }
}