use variable::{Variable, VariableInner};
use function::{Function, FunctionInner};
use axis::Axis;
use shape::Shape;
use std::borrow::Borrow;
use std::ptr;
use std::ffi::CStr;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>
  #include <iostream>

  using namespace CNTK;
  using namespace std;
}}

pub fn transpose_axes<T: Into<Variable>>(x: T, axis1: &Axis, axis2: &Axis) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let a1payload = axis1.payload;
    let a2payload = axis2.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", a1payload as "Axis", a2payload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return TransposeAxes(xpayload, a1payload, a2payload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function { payload }
}

pub fn dropout<T: Into<Variable>>(x: T, dropout_rate: f64) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", dropout_rate as "double", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Dropout(xpayload, dropout_rate);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function { payload }
}

// TODO: Make this more friendly
pub fn splice(variables: &[&Variable], axis: &Axis) -> Function {
    let data: Vec<Variable> = variables.iter().map(|&x| x.clone()).collect();
    let data_ptr = data.as_ptr();
    let data_size = data.len();
    let apayload = axis.payload;
    Function { payload: unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([data_ptr as "Variable*", data_size as "size_t", apayload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Splice(vector<Variable>(data_ptr, data_ptr + data_size), apayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    }}
}

pub fn reshape<T: Into<Variable>>(x: T, shape: &Shape) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let spayload = shape.payload;
    Function { payload: unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", spayload as "NDShape", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Reshape(xpayload, spayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    }}
}

pub fn slice<T: Into<Variable>>(x: T, axis: &[&Axis], begin_index: &[i32], end_index: &[i32]) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    assert_eq!(axis.len(), begin_index.len());
    assert_eq!(axis.len(), end_index.len());
    let len = axis.len();
    let adata: Vec<Axis> = axis.iter().map(|&x| x.clone()).collect();
    let adata_ptr = adata.as_ptr();
    let bdata_ptr = begin_index.as_ptr();
    let edata_ptr = end_index.as_ptr();

    Function { payload: unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", adata_ptr as "Axis*", len as "size_t", bdata_ptr as "int*", edata_ptr as "int*", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Slice(xpayload,
                             vector<Axis>(adata_ptr, adata_ptr + len),
                             vector<int>(bdata_ptr, bdata_ptr + len),
                             vector<int>(edata_ptr, edata_ptr + len));
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    }}
}

pub fn named_alias<T: Into<Variable>>(x: T, name: &str) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let name_ptr = name.as_ptr();
    let name_len = name.len();
    Function { payload: unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", name_ptr as "char*", name_len as "size_t", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try{
                string name(name_ptr, name_ptr + name_len);
                wstring wname;
                wname.assign(name.begin(), name.end());
                return Alias(xpayload, wname);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    }}
}

pub fn past_value<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return PastValue(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn future_value<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return FutureValue(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn past_value_with_init<T: Into<Variable>, U: Into<Variable>>(x: T, initial: U) -> Function {
    let xv = x.into();
    let iv = initial.into();
    let xpayload = xv.payload;
    let ipayload = iv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ipayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return PastValue(xpayload, ipayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn future_value_with_init<T: Into<Variable>, U: Into<Variable>>(x: T, initial: U) -> Function {
    let xv = x.into();
    let iv = initial.into();
    let xpayload = xv.payload;
    let ipayload = iv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ipayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return FutureValue(xpayload, ipayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn first<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Sequence::First(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn last<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Sequence::Last(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

/* unary ops begin here */


pub fn negate<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Negate(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn sigmoid<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Sigmoid(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn tanh<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Tanh(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn asin<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Asin(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn sin<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Sin(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn acos<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Acos(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn cos<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Cos(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn cosh<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Cosh(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn sinh<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Sinh(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn relu<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ReLU(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn exp<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Exp(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn log<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Log(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn square<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Square(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn sqrt<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Sqrt(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn round<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Round(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn floor<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Floor(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn ceil<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Ceil(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn abs<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Abs(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn reciprocal<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Reciprocal(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn softmax<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Softmax(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn hardmax<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Hardmax(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn transpose<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Transpose(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn to_batch<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ToBatch(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn alias<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Alias(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn stop_gradient<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return StopGradient(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn elu<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ELU(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn leaky_relu<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return LeakyReLU(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn softplus<T: Into<Variable>>(x: T) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Softplus(xpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}


/* unary ops end here */

/* binary ops begin here */

pub fn plus<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Plus(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn minus<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Minus(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn log_add_exp<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return LogAddExp(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn pow<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Pow(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn element_times<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ElementTimes(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn element_divide<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ElementDivide(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn equal<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Equal(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn not_equal<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return NotEqual(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn less<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Less(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn less_equal<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return LessEqual(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn greater<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Greater(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn greater_equal<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return GreaterEqual(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn times<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p : *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Times(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
            let msg = CStr::from_ptr(error_p).to_str().unwrap();
            panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn transpose_times<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return TransposeTimes(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn cosine_distance<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return CosineDistance(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn binary_cross_entropy<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return BinaryCrossEntropy(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn squared_error<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return SquaredError(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn cross_entropy_with_softmax<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return CrossEntropyWithSoftmax(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn classification_error<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ClassificationError(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}


/* binary ops end here */

/* unary axis ops start here */


pub fn softmax_with_axis<T: Into<Variable>>(x: T, axis: &Axis) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let apayload = axis.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", apayload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Softmax(xpayload, apayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn reduce_sum<T: Into<Variable>>(x: T, axis: &Axis) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let apayload = axis.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", apayload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ReduceSum(xpayload, apayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn reduce_log_sum<T: Into<Variable>>(x: T, axis: &Axis) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let apayload = axis.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", apayload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ReduceLogSum(xpayload, apayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn reduce_mean<T: Into<Variable>>(x: T, axis: &Axis) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let apayload = axis.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", apayload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ReduceMean(xpayload, apayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn reduce_max<T: Into<Variable>>(x: T, axis: &Axis) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let apayload = axis.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", apayload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ReduceMax(xpayload, apayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn reduce_min<T: Into<Variable>>(x: T, axis: &Axis) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let apayload = axis.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", apayload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ReduceMin(xpayload, apayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn reduce_prod<T: Into<Variable>>(x: T, axis: &Axis) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let apayload = axis.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", apayload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ReduceProd(xpayload, apayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn argmax<T: Into<Variable>>(x: T, axis: &Axis) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let apayload = axis.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", apayload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Argmax(xpayload, apayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn argmin<T: Into<Variable>>(x: T, axis: &Axis) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let apayload = axis.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", apayload as "Axis", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Argmin(xpayload, apayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}



/* unary axis ops end here */

/* random ops */
pub fn normal_random_like<T: Into<Variable>>(x: T, mean: f64, scale: f64) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mean as "double", scale as "double", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return NormalRandomLike(xpayload, mean, scale);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn bernoulli_random_like<T: Into<Variable>>(x: T, mean: f64) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", mean as "double", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return BernoulliRandomLike(xpayload, mean);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn uniform_random_like<T: Into<Variable>>(x: T, low: f64, high: f64) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", low as "double", high as "double", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return UniformRandomLike(xpayload, low, high);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn gumbel_random_like<T: Into<Variable>>(x: T, loc: f64, scale: f64) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", loc as "double", scale as "double", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return GumbelRandomLike(xpayload, loc, scale);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

/* random ops end */

/* convolution */
pub fn convolution<T: Into<Variable>, U: Into<Variable>>(convmap: T, y: U, strides: &Shape) -> Function {
    let convmapv = convmap.into();
    let convmappayload = convmapv.payload;
    let yv = y.into();
    let ypayload = yv.payload;
    let spayload = strides.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([convmappayload as "Variable", ypayload as "Variable", spayload as "NDShape", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Convolution(convmappayload, ypayload, spayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn max_pooling<T: Into<Variable>>(x: T, window_shape: &Shape, strides: &Shape) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let spayload = window_shape.payload;
    let stpayload = strides.payload;
    Function { payload: unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", spayload as "NDShape", stpayload as "NDShape", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Pooling(xpayload, PoolingType::Max, spayload, stpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    }}
}

pub fn avg_pooling<T: Into<Variable>>(x: T, window_shape: &Shape, strides: &Shape) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let spayload = window_shape.payload;
    let stpayload = strides.payload;
    Function { payload: unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", spayload as "NDShape", stpayload as "NDShape", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Pooling(xpayload, PoolingType::Average, spayload, stpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    }}
}

pub fn clip<T: Into<Variable>, U: Into<Variable>, V: Into<Variable>>(x: T, min: U, max: V) -> Function {
    let xv = x.into();
    let xpayload = xv.payload;
    let minv = min.into();
    let minpayload = minv.payload;
    let maxv = max.into();
    let maxpayload = maxv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", minpayload as "Variable", maxpayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Clip(xpayload, minpayload, maxpayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn nce_loss<T: Into<Variable>, U: Into<Variable>, V: Into<Variable>, W: Into<Variable>, X: Into<Variable>>(weights: T, biases: U, inputs: V, labels: W, noise_weights: X, num_samples: usize) -> Function {
    let wv = weights.into();
    let bv = biases.into();
    let iv = inputs.into();
    let lv = labels.into();
    let nv = noise_weights.into();
    let wvp = wv.payload;
    let bvp = bv.payload;
    let ivp = iv.payload;
    let lvp = lv.payload;
    let nvp = nv.payload;
    Function { payload: unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([wvp as "Variable", bvp as "Variable", ivp as "Variable", lvp as "Variable", nvp as "Variable", num_samples as "size_t", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return NCELoss(wvp, bvp, ivp, lvp, Constant(nvp), num_samples);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    }}
}

pub fn broadcast_as<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.payload;
    let ypayload: VariableInner = yv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Sequence::BroadcastAs(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn unpack<T: Into<Variable>>(x: T, padding_value: f32) -> Function {
    let xv = x.into();
    let xpayload: VariableInner = xv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", padding_value as "float", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return Sequence::Unpack(xpayload, padding_value, true);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}

pub fn to_sequence_like<T: Into<Variable>, U: Into<Variable>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.payload;
    let ypayload: VariableInner = yv.payload;
    let payload = unsafe {
        let mut error_p: *mut i8 = ptr::null_mut();
        let payload = cpp!([xpayload as "Variable", ypayload as "Variable", mut error_p as "char*"] -> FunctionInner as "FunctionPtr" {
            try {
                return ToSequenceLike(xpayload, ypayload);
            } catch (std::exception& e) {
                auto what = e.what();
                error_p = new char[strlen(what)+1];
                strcpy(error_p, what);
                return nullptr;
            }
        });
        if !error_p.is_null() {
             let msg = CStr::from_ptr(error_p).to_str().unwrap();
             panic!("{}", msg);
        }
        payload
    };
    Function {payload}
}