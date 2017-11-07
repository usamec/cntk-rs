use variable::{Variable, VariableInner};

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

pub fn elem_times(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return ElementTimes(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn times(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return Times(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn cross_entropy_with_softmax(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return CrossEntropyWithSoftmax(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn squared_error(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return SquaredError(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn classification_error(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return ClassificationError(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn tanh(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Tanh(xpayload);
        })
    };
    Variable {payload}
}

pub fn reduce_sum_all(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return ReduceSum(xpayload, Axis::AllAxes());
        })
    };
    Variable {payload}
}

pub fn named_alias(x: &Variable, name: &str) -> Variable {
    let xpayload = x.payload;
    let name_ptr = name.as_ptr();
    let name_len = name.len();
    Variable { payload: unsafe {
        cpp!([xpayload as "Variable", name_ptr as "char*", name_len as "size_t"] -> VariableInner as "Variable" {
                string name(name_ptr, name_ptr + name_len);
                wstring wname;
                wname.assign(name.begin(), name.end());
                return Alias(xpayload, wname);
            })
    }}
}

pub fn past_value(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return PastValue(xpayload);
        })
    };
    Variable {payload}
}

pub fn last(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Sequence::Last(xpayload);
        })
    };
    Variable {payload}
}