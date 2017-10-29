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
}