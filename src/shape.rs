use std::borrow::Borrow;

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

    pub fn new<K: Borrow<Vec<usize>>>(data: K) -> Shape {
        let datab = data.borrow();
        let datab_ptr = datab.as_ptr();
        let datab_size = datab.len();
        Shape {payload: unsafe {
            cpp!([datab_ptr as "size_t*", datab_size as "size_t"] -> ShapeInner as "NDShape" {
                vector<size_t> shape_vec(datab_ptr, datab_ptr + datab_size);
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

    pub fn to_vec(&self) -> Vec<usize> {
        let mut ret = Vec::new();
        for i in 0..self.rank() {
            ret.push(self.get(i));
        }
        ret
    }

    pub fn to_vec_reversed(&self) -> Vec<usize> {
        let mut ret = Vec::new();
        for i in 0..self.rank() {
            ret.push(self.get(i));
        }
        ret.reverse();
        ret
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