use variable::{Variable, VariableInner, IntoVariable, VariableOrRef};
use function::{Function, FunctionInner};
use axis::Axis;
use shape::Shape;
use std::borrow::Borrow;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>
  #include <iostream>

  using namespace CNTK;
  using namespace std;
}}

pub fn transpose_axes(x: &Variable, axis1: &Axis, axis2: &Axis) -> Variable {
    let xpayload = x.payload;
    let a1payload = axis1.payload;
    let a2payload = axis2.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", a1payload as "Axis", a2payload as "Axis"] -> VariableInner as "Variable" {
            return TransposeAxes(xpayload, a1payload, a2payload);
        })
    };
    Variable { payload }
}

pub fn dropout(x: &Variable, dropout_rate: f64) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", dropout_rate as "double"] -> VariableInner as "Variable" {
            return Dropout(xpayload, dropout_rate);
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

pub fn splice(variables: &[&Variable], axis: &Axis) -> Variable {
    let data: Vec<Variable> = variables.iter().map(|&x| x.clone()).collect();
    let data_ptr = data.as_ptr();
    let data_size = data.len();
    let apayload = axis.payload;
    Variable { payload: unsafe {
        cpp!([data_ptr as "Variable*", data_size as "size_t", apayload as "Axis"] -> VariableInner as "Variable" {
                return Splice(vector<Variable>(data_ptr, data_ptr + data_size), apayload);
            })
    }}
}

pub fn reshape(x: &Variable, shape: &Shape) -> Variable {
    let xpayload = x.payload;
    let spayload = shape.payload;
    Variable { payload: unsafe {
        cpp!([xpayload as "Variable", spayload as "NDShape"] -> VariableInner as "Variable" {
                return Reshape(xpayload, spayload);
            })
    }}
}

pub fn slice(x: &Variable, axis: &[&Axis], begin_index: &[i32], end_index: &[i32]) -> Variable {
    let xpayload = x.payload;
    assert_eq!(axis.len(), begin_index.len());
    assert_eq!(axis.len(), end_index.len());
    let len = axis.len();
    let adata: Vec<Axis> = axis.iter().map(|&x| x.clone()).collect();
    let adata_ptr = adata.as_ptr();
    let bdata_ptr = begin_index.as_ptr();
    let edata_ptr = end_index.as_ptr();

    Variable { payload: unsafe {
        cpp!([xpayload as "Variable", adata_ptr as "Axis*", len as "size_t", bdata_ptr as "int*", edata_ptr as "int*"] -> VariableInner as "Variable" {
                return Slice(xpayload,
                             vector<Axis>(adata_ptr, adata_ptr + len),
                             vector<int>(bdata_ptr, bdata_ptr + len),
                             vector<int>(edata_ptr, edata_ptr + len));
            })
    }}
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

pub fn future_value(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return FutureValue(xpayload);
        })
    };
    Variable {payload}
}

pub fn first(x: &Variable) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable"] -> VariableInner as "Variable" {
            return Sequence::First(xpayload);
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

pub fn relu(x: &Variable) -> Variable {
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

pub fn plus<TT: VariableOrRef, T: IntoVariable<TT>, UU: VariableOrRef, U: IntoVariable<UU>>(x: T, y: U) -> Function {
    let xv = x.into();
    let yv = y.into();
    let xpayload: VariableInner = xv.borrow().payload;
    let ypayload: VariableInner = yv.borrow().payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", ypayload as "Variable"] -> FunctionInner as "FunctionPtr" {
            try {
                return Plus(xpayload, ypayload);
            } catch (std::exception& e) {
                printf("Plus throw an exception %s\n", e.what());
                throw e;
            }
        })
    };
    Function {payload}
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

/* unary axis ops start here */
pub fn softmax_with_axis(x: &Variable, axis: &Axis) -> Variable {
    let xpayload = x.payload;
    let apayload = axis.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", apayload as "Axis"] -> VariableInner as "Variable" {
            return Softmax(xpayload, apayload);
        })
    };
    Variable {payload}
}

pub fn reduce_sum(x: &Variable, axis: &Axis) -> Variable {
    let xpayload = x.payload;
    let apayload = axis.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", apayload as "Axis"] -> VariableInner as "Variable" {
            return ReduceSum(xpayload, apayload);
        })
    };
    Variable {payload}
}

pub fn reduce_log_sum(x: &Variable, axis: &Axis) -> Variable {
    let xpayload = x.payload;
    let apayload = axis.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", apayload as "Axis"] -> VariableInner as "Variable" {
            return ReduceLogSum(xpayload, apayload);
        })
    };
    Variable {payload}
}

pub fn reduce_mean(x: &Variable, axis: &Axis) -> Variable {
    let xpayload = x.payload;
    let apayload = axis.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", apayload as "Axis"] -> VariableInner as "Variable" {
            return ReduceMean(xpayload, apayload);
        })
    };
    Variable {payload}
}

pub fn reduce_max(x: &Variable, axis: &Axis) -> Variable {
    let xpayload = x.payload;
    let apayload = axis.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", apayload as "Axis"] -> VariableInner as "Variable" {
            return ReduceMax(xpayload, apayload);
        })
    };
    Variable {payload}
}

pub fn reduce_min(x: &Variable, axis: &Axis) -> Variable {
    let xpayload = x.payload;
    let apayload = axis.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", apayload as "Axis"] -> VariableInner as "Variable" {
            return ReduceMin(xpayload, apayload);
        })
    };
    Variable {payload}
}

pub fn reduce_prod(x: &Variable, axis: &Axis) -> Variable {
    let xpayload = x.payload;
    let apayload = axis.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", apayload as "Axis"] -> VariableInner as "Variable" {
            return ReduceProd(xpayload, apayload);
        })
    };
    Variable {payload}
}

pub fn argmax(x: &Variable, axis: &Axis) -> Variable {
    let xpayload = x.payload;
    let apayload = axis.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", apayload as "Axis"] -> VariableInner as "Variable" {
            return Argmax(xpayload, apayload);
        })
    };
    Variable {payload}
}

pub fn argmin(x: &Variable, axis: &Axis) -> Variable {
    let xpayload = x.payload;
    let apayload = axis.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", apayload as "Axis"] -> VariableInner as "Variable" {
            return Argmin(xpayload, apayload);
        })
    };
    Variable {payload}
}
/* unary axis ops end here */

/* random ops */
pub fn normal_random_like(x: &Variable, mean: f64, scale: f64) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", mean as "double", scale as "double"] -> VariableInner as "Variable" {
            return NormalRandomLike(xpayload, mean, scale);
        })
    };
    Variable {payload}
}

pub fn bernoulli_random_like(x: &Variable, mean: f64) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", mean as "double"] -> VariableInner as "Variable" {
            return BernoulliRandomLike(xpayload, mean);
        })
    };
    Variable {payload}
}

pub fn uniform_random_like(x: &Variable, low: f64, high: f64) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", low as "double", high as "double"] -> VariableInner as "Variable" {
            return UniformRandomLike(xpayload, low, high);
        })
    };
    Variable {payload}
}

pub fn gumbel_random_like(x: &Variable, loc: f64, scale: f64) -> Variable {
    let xpayload = x.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", loc as "double", scale as "double"] -> VariableInner as "Variable" {
            return GumbelRandomLike(xpayload, loc, scale);
        })
    };
    Variable {payload}
}

/* random ops end */

/* convolution */
pub fn convolution(convmap: &Variable, y: &Variable, strides: &Shape) -> Variable {
    let convmappayload = convmap.payload;
    let ypayload = y.payload;
    let spayload = strides.payload;
    let payload = unsafe {
        cpp!([convmappayload as "Variable", ypayload as "Variable", spayload as "NDShape"] -> VariableInner as "Variable" {
            try {
                return Convolution(convmappayload, ypayload, spayload);
            } catch (std::exception& e) {
                printf("Convolution op threw an exception %s\n", e.what());
                throw e;
            }
        })
    };
    Variable {payload}
}

pub fn max_pooling(x: &Variable, window_shape: &Shape, strides: &Shape) -> Variable {
    let xpayload = x.payload;
    let spayload = window_shape.payload;
    let stpayload = strides.payload;
    Variable { payload: unsafe {
        cpp!([xpayload as "Variable", spayload as "NDShape", stpayload as "NDShape"] -> VariableInner as "Variable" {
            return Pooling(xpayload, PoolingType::Max, spayload, stpayload);
        })
    }}
}

pub fn avg_pooling(x: &Variable, window_shape: &Shape, strides: &Shape) -> Variable {
    let xpayload = x.payload;
    let spayload = window_shape.payload;
    let stpayload = strides.payload;
    Variable { payload: unsafe {
        cpp!([xpayload as "Variable", spayload as "NDShape", stpayload as "NDShape"] -> VariableInner as "Variable" {
                return Pooling(xpayload, PoolingType::Average, spayload, stpayload);
            })
    }}
}

pub fn clip(x: &Variable, min: &Variable, max: &Variable) -> Variable {
    let xpayload = x.payload;
    let minpayload = min.payload;
    let maxpayload = max.payload;
    let payload = unsafe {
        cpp!([xpayload as "Variable", minpayload as "Variable", maxpayload as "Variable"] -> VariableInner as "Variable" {
            return Clip(xpayload, minpayload, maxpayload);
        })
    };
    Variable {payload}
}