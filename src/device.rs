cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

type DeviceDescriptorInner = [u32; 2usize];

#[derive(Copy, Clone)]
pub struct DeviceDescriptor {
    pub(super) payload: DeviceDescriptorInner
}

impl DeviceDescriptor {
    pub fn cpu() -> DeviceDescriptor {
        DeviceDescriptor {
            payload: unsafe {
                    cpp!([] -> DeviceDescriptorInner as "DeviceDescriptor" {
                    return DeviceDescriptor::CPUDevice();
                })
            }
        }
    }
}

pub fn set_max_num_cpu_threads(num: usize) {
    unsafe {
        cpp!([num as "size_t"] {
            SetMaxNumCPUThreads(num);
        })
    }
}