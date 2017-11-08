use variable::{Variable, VariableInner};

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

/* unary ops begin here */

pub fn negate(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Negate(xpayload);
        })
    };
    Variable {payload}
}

pub fn sigmoid(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Sigmoid(xpayload);
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

pub fn asin(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Asin(xpayload);
        })
    };
    Variable {payload}
}

pub fn sin(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Sin(xpayload);
        })
    };
    Variable {payload}
}

pub fn acos(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Acos(xpayload);
        })
    };
    Variable {payload}
}

pub fn cos(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Cos(xpayload);
        })
    };
    Variable {payload}
}

pub fn cosh(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Cosh(xpayload);
        })
    };
    Variable {payload}
}

pub fn sinh(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Sinh(xpayload);
        })
    };
    Variable {payload}
}

pub fn re_lu(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return ReLU(xpayload);
        })
    };
    Variable {payload}
}

pub fn exp(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Exp(xpayload);
        })
    };
    Variable {payload}
}

pub fn log(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Log(xpayload);
        })
    };
    Variable {payload}
}

pub fn square(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Square(xpayload);
        })
    };
    Variable {payload}
}

pub fn sqrt(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Sqrt(xpayload);
        })
    };
    Variable {payload}
}

pub fn round(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Round(xpayload);
        })
    };
    Variable {payload}
}

pub fn floor(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Floor(xpayload);
        })
    };
    Variable {payload}
}

pub fn ceil(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Ceil(xpayload);
        })
    };
    Variable {payload}
}

pub fn abs(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Abs(xpayload);
        })
    };
    Variable {payload}
}

pub fn reciprocal(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Reciprocal(xpayload);
        })
    };
    Variable {payload}
}

pub fn softmax(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Softmax(xpayload);
        })
    };
    Variable {payload}
}

pub fn hardmax(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Hardmax(xpayload);
        })
    };
    Variable {payload}
}

pub fn transpose(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Transpose(xpayload);
        })
    };
    Variable {payload}
}

pub fn to_batch(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return ToBatch(xpayload);
        })
    };
    Variable {payload}
}

pub fn alias(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Alias(xpayload);
        })
    };
    Variable {payload}
}

pub fn stop_gradient(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return StopGradient(xpayload);
        })
    };
    Variable {payload}
}

pub fn elu(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return ELU(xpayload);
        })
    };
    Variable {payload}
}

pub fn leaky_re_lu(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return LeakyReLU(xpayload);
        })
    };
    Variable {payload}
}

pub fn softplus(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Softplus(xpayload);
        })
    };
    Variable {payload}
}

/* unary ops end here */

/* binary ops begin here */

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

pub fn minus(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return Minus(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn log_add_exp(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return LogAddExp(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn pow(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return Pow(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn element_times(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return ElementTimes(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn element_divide(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return ElementDivide(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn equal(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return Equal(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn not_equal(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return NotEqual(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn less(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return Less(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn less_equal(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return LessEqual(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn greater(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return Greater(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn greater_equal(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return GreaterEqual(xpayload, ypayload);
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

pub fn transpose_times(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return TransposeTimes(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn cosine_distance(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return CosineDistance(xpayload, ypayload);
        })
    };
    Variable {payload}
}

pub fn binary_cross_entropy(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return BinaryCrossEntropy(xpayload, ypayload);
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

pub fn classification_error(x: &Variable, y: &Variable) -> Variable {
    let xpayload = x.payload;
    let ypayload = y.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> VariableInner as "Variable" {
            return ClassificationError(xpayload, ypayload);
        })
    };
    Variable { payload }
}

/* binary ops end here */