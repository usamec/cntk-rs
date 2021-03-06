cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

pub(super) type AxisInner = [u64; 6usize];

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

    pub fn default_batch_axis() -> Axis {
        Axis {
            payload: unsafe {
                cpp!([] -> AxisInner as "Axis" {
                    return Axis::DefaultBatchAxis();
                })
            }
        }
    }

    pub fn named_dynamic(name: &str) -> Axis {
        let name_ptr = name.as_ptr();
        let name_len = name.len();
        Axis {
            payload: unsafe {
                cpp!([name_ptr as "char*", name_len as "size_t"] -> AxisInner as "Axis" {
                    string name(name_ptr, name_ptr + name_len);
                    wstring wname;
                    wname.assign(name.begin(), name.end());
                    return Axis(wname);
                })
            }
        }
    }

    pub fn all_static() -> Axis {
        Axis {
            payload: unsafe {
                cpp!([] -> AxisInner as "Axis" {
                    return Axis::AllStaticAxes();
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
