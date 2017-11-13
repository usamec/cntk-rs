use variable::{Variable};
use std::borrow::Borrow;

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

type DoubleParameterScheduleInner = [u64; 9usize];

pub struct DoubleParameterSchedule {
    pub(super) payload: DoubleParameterScheduleInner
}

impl DoubleParameterSchedule {
    pub fn constant(x: f64) -> DoubleParameterSchedule {
        DoubleParameterSchedule {payload: unsafe {
            cpp!([x as "double"] -> DoubleParameterScheduleInner as "TrainingParameterSchedule<double>" {
                return TrainingParameterPerSampleSchedule(x);
            })
        }}
    }
}

impl Drop for DoubleParameterSchedule {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "TrainingParameterSchedule<double>"] {
                payload.~TrainingParameterSchedule();
            })
        };
    }
}

type LearnerInner = [u64; 2usize];

#[derive(Debug)]
pub struct Learner {
    pub(super) payload: LearnerInner
}

impl Learner {
    pub fn sgd<T: Borrow<Variable>>(parameters: &[T], learning_rate_schedule: &DoubleParameterSchedule) -> Learner {
        check_parameters(parameters);

        let data: Vec<Variable> = parameters.iter().map(|x| x.borrow().clone()).collect();
        let data_ptr = data.as_ptr();
        let data_size = data.len();
        let schedule = learning_rate_schedule.payload;
        Learner { payload: unsafe {
            cpp!([data_ptr as "Parameter*", data_size as "size_t", schedule as "TrainingParameterSchedule<double>"] -> LearnerInner as "LearnerPtr" {
                return SGDLearner(vector<Parameter>(data_ptr, data_ptr + data_size), schedule);
            })
        }}
    }

    pub fn momentum_sgd<T: Borrow<Variable>>(parameters: &[T], learning_rate_schedule: &DoubleParameterSchedule, momentum_schedule: &DoubleParameterSchedule) -> Learner {
        check_parameters(parameters);

        let data: Vec<Variable> = parameters.iter().map(|x| x.borrow().clone()).collect();
        let data_ptr = data.as_ptr();
        let data_size = data.len();
        let schedule = learning_rate_schedule.payload;
        let mschedule = momentum_schedule.payload;
        Learner { payload: unsafe {
            cpp!([data_ptr as "Parameter*", data_size as "size_t", schedule as "TrainingParameterSchedule<double>", mschedule as "TrainingParameterSchedule<double>"] -> LearnerInner as "LearnerPtr" {
                return MomentumSGDLearner(vector<Parameter>(data_ptr, data_ptr + data_size), schedule, mschedule);
            })
        }}
    }

    pub fn adam<T: Borrow<Variable>>(parameters: &[T], learning_rate_schedule: &DoubleParameterSchedule, momentum_schedule: &DoubleParameterSchedule) -> Learner {
        check_parameters(parameters);

        let data: Vec<Variable> = parameters.iter().map(|x| x.borrow().clone()).collect();
        let data_ptr = data.as_ptr();
        let data_size = data.len();
        let schedule = learning_rate_schedule.payload;
        let mschedule = momentum_schedule.payload;
        Learner { payload: unsafe {
            cpp!([data_ptr as "Parameter*", data_size as "size_t", schedule as "TrainingParameterSchedule<double>", mschedule as "TrainingParameterSchedule<double>"] -> LearnerInner as "LearnerPtr" {
                return AdamLearner(vector<Parameter>(data_ptr, data_ptr + data_size), schedule, mschedule);
            })
        }}
    }
}

impl Drop for Learner {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "LearnerPtr"] {
                payload.~LearnerPtr();
            })
        };
    }
}

fn check_parameters<T: Borrow<Variable>>(parameters: &[T]) {
    for parameter in parameters {
        assert!(parameter.borrow().is_parameter());
    }
}