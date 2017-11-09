cpp! {{
  #include <CNTKLibrary.h>
  #include <cstdio>
  #include <vector>

  using namespace CNTK;
  using namespace std;
}}

pub(super) type ShapeInner = [u64; 3usize];

pub struct Shape {
    pub(super) payload: ShapeInner
}

impl Shape {
    pub fn scalar() -> Shape {
        Shape {payload: unsafe {
            cpp!([] -> ShapeInner as "NDShape" {
                return NDShape();
            })
        }}
    }

    pub fn from_slice(data: &[usize]) -> Shape {
        let data_ptr = data.as_ptr();
        let data_size = data.len();
        Shape {payload: unsafe {
            cpp!([data_ptr as "size_t*", data_size as "size_t"] -> ShapeInner as "NDShape" {
                vector<size_t> shape_vec(data_ptr, data_ptr + data_size);
                return NDShape(shape_vec);
            })
        }}
    }

    pub fn total_size(&self) -> usize {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "NDShape"] -> usize as "size_t" {
                return payload.TotalSize();
            })
        }
    }

    pub fn rank(&self) -> usize {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "NDShape"] -> usize as "size_t" {
                return payload.Rank();
            })
        }
    }

    pub fn get(&self, axis: usize) -> usize {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "NDShape", axis as "size_t"] -> usize as "size_t" {
                return payload[axis];
            })
        }
    }

    pub fn append_shape(&self, shape: &Shape) -> Shape {
        let payload = self.payload;
        let spayload = shape.payload;
        Shape {payload: unsafe {
            cpp!([payload as "NDShape", spayload as "NDShape"] -> ShapeInner as "NDShape" {
                return payload.AppendShape(spayload);
            })
        }}
    }
}

impl Drop for Shape {
    fn drop(&mut self) {
        let payload = self.payload;
        unsafe {
            cpp!([payload as "NDShape"] {
                payload.~NDShape();
            })
        };
    }
}