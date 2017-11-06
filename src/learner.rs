use variable::{Variable};

cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

type LearnerInner = [u64; 2usize];

#[derive(Debug)]
pub struct Learner {
    pub(super) payload: LearnerInner
}

impl Learner {
    pub fn sgd(parameters: &[&Variable]) -> Learner {
        // TODO: assert all parameters are really parameters
        let data: Vec<Variable> = parameters.iter().map(|&x| x.clone()).collect();
        let data_ptr = data.as_ptr();
        let data_size = data.len();
        Learner { payload: unsafe {
            cpp!([data_ptr as "Parameter*", data_size as "size_t"] -> LearnerInner as "LearnerPtr" {
                return SGDLearner(vector<Parameter>(data_ptr, data_ptr + data_size), TrainingParameterPerSampleSchedule(0.001));
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