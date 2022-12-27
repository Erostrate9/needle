#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
// #define TILE 4
#define TILE 16
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
typedef ssize_t ptrdiff_t;

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides




__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid < size){
    size_t idx = 0, id = gid;
    for (size_t i = shape.size; i>0; --i){
        idx += (id % shape.data[i-1]) * strides.data[i-1];
        id /= shape.data[i-1];
    }
    out[gid] = a[offset + idx];
  }
  /// END YOUR SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the EwiseSetitem operation.
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid < size){
    size_t idx = 0, id = gid;
    for (size_t i = shape.size; i>0; --i){
        idx += (id % shape.data[i-1]) * strides.data[i-1];
        id /= shape.data[i-1];
    }
    out[offset + idx] = a[gid];
  }
  /// END YOUR SOLUTION
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
    CudaDims dim = CudaOneDim(out->size);
    EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}


__global__ void ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the ScalarSetitem operation.
   *
   * Args:
   *   size: number of elements to write in out array (note that this will not be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  if (gid < size){
    size_t idx = 0, id = gid;
    for (size_t i = shape.size; i>0; --i){
        idx += (id % shape.data[i-1]) * strides.data[i-1];
        id /= shape.data[i-1];
    }
    out[offset + idx] = val;
  }
  /// END YOUR SOLUTION
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
    CudaDims dim = CudaOneDim(out->size);
    ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
// macros BEGIN
#define KERNEL_LEFT size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid]

#define INIT_DIM CudaDims dim = CudaOneDim(out->size)

#define EWISE_BINARY_KERNEL <<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size)

#define EWISE_UNARY_KERNEL <<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size)

#define SCALAR_BINARY_KERNEL <<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size)

// macros END

//  *   - EwiseMul, ScalarMul
__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    KERNEL_LEFT = a[gid] * b[gid];
}
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Multiply together two CUDA array
   */
  INIT_DIM;
  EwiseMulKernel EWISE_BINARY_KERNEL;
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    KERNEL_LEFT = a[gid] * val;
}
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  INIT_DIM;
  ScalarMulKernel SCALAR_BINARY_KERNEL;
}

//  *   - EwiseDiv, ScalarDiv
__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    KERNEL_LEFT = a[gid] / b[gid];
}
void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  INIT_DIM;
  EwiseDivKernel EWISE_BINARY_KERNEL;
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    KERNEL_LEFT = a[gid] / val;
}
void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  INIT_DIM;
  ScalarDivKernel SCALAR_BINARY_KERNEL;
}
//  *   - ScalarPower
__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    KERNEL_LEFT = powf(a[gid], val);
}
void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  INIT_DIM;
  ScalarPowerKernel SCALAR_BINARY_KERNEL;
}
//  *   - EwiseMaximum, ScalarMaximum
__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    KERNEL_LEFT = max(a[gid], b[gid]);
}
void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  INIT_DIM;
  EwiseMaximumKernel EWISE_BINARY_KERNEL;
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    KERNEL_LEFT = max(a[gid], val);
}
void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  INIT_DIM;
  ScalarMaximumKernel SCALAR_BINARY_KERNEL;
}
//  *   - EwiseEq, ScalarEq
__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    KERNEL_LEFT = (a[gid] == b[gid]);
}
void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  INIT_DIM;
  EwiseEqKernel EWISE_BINARY_KERNEL;
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    KERNEL_LEFT = (a[gid] == val);
}
void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  INIT_DIM;
  ScalarEqKernel SCALAR_BINARY_KERNEL;
}
//  *   - EwiseGe, ScalarGe
__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    KERNEL_LEFT = (a[gid] >= b[gid]);
}
void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  INIT_DIM;
  EwiseGeKernel EWISE_BINARY_KERNEL;
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    KERNEL_LEFT = (a[gid] >= val);
}
void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  INIT_DIM;
  ScalarGeKernel SCALAR_BINARY_KERNEL;
}
//  *   - EwiseLog
__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
    KERNEL_LEFT = logf(a[gid]);
}
void EwiseLog(const CudaArray& a, CudaArray* out) {
  INIT_DIM;
  EwiseLogKernel EWISE_UNARY_KERNEL;
}
//  *   - EwiseExp
__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
    KERNEL_LEFT = expf(a[gid]);
}
void EwiseExp(const CudaArray& a, CudaArray* out) {
  INIT_DIM;
  EwiseExpKernel EWISE_UNARY_KERNEL;
}
//  *   - EwiseTanh
__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
    KERNEL_LEFT = tanhf(a[gid]);
}
void EwiseTanh(const CudaArray& a, CudaArray* out) {
  INIT_DIM;
  EwiseTanhKernel EWISE_UNARY_KERNEL;
}

//  *   - EwiseSin
__global__ void EwiseSinKernel(const scalar_t* a, scalar_t* out, size_t size) {
    KERNEL_LEFT = sinf(a[gid]);
}
void EwiseSin(const CudaArray& a, CudaArray* out) {
  INIT_DIM;
  EwiseSinKernel EWISE_UNARY_KERNEL;
}

//  *   - EwiseCos
__global__ void EwiseCosKernel(const scalar_t* a, scalar_t* out, size_t size) {
    KERNEL_LEFT = cosf(a[gid]);
}
void EwiseCos(const CudaArray& a, CudaArray* out) {
  INIT_DIM;
  EwiseCosKernel EWISE_UNARY_KERNEL;
}
/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////


__global__ void MatmulKernel_naive(const scalar_t* a, const scalar_t* b, scalar_t* out, const size_t M, const size_t N, const size_t P) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid < M*P){
        size_t row = gid / P;
        size_t col = gid % P;
        out[gid] = 0.0f;
        for (size_t k=0; k<N; ++k){
            out[gid] += a[row*N + k] * b[k * P + col];
        }
    }
}

__global__ void MatmulKernel_tiled(const scalar_t* a, const scalar_t* b, scalar_t* out, const size_t M, const size_t N, const size_t P) {
    size_t bidx = blockIdx.x, bidy = blockIdx.y,
           tidx = threadIdx.x, tidy = threadIdx.y;
    int x_range = static_cast<int>(bidx + 1) * TILE - M,
        y_range = static_cast<int>(bidy + 1) * TILE - P;
    if (x_range > 0) {
        a -= x_range * N;
        out -= x_range * P;
    }
    if (y_range > 0) {
        b -= y_range;
        out -= y_range;
    }
    a += bidx * TILE * N;
    b += bidy * TILE;
    out += (bidx * TILE) * P + (bidy * TILE);
    __shared__ scalar_t smemA[TILE][TILE], smemB[TILE][TILE];
    scalar_t accumu = 0.0f;
    for (int i = 0; i < N; i += TILE) {
        smemA[tidx][tidy] = (tidy + i < N) ? a[(tidx)*N + (tidy + i)] : 0.0f;
        smemB[tidx][tidy] = (tidx + i < N) ? b[(tidx + i) * P + tidy] : 0.0f;
        __syncthreads();
        for (int j = 0; j < TILE; j++) {
            accumu += smemA[tidx][j] * smemB[j][tidy];
        }
        __syncthreads();
    }
    out[tidx * P + tidy] = accumu;
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN YOUR SOLUTION
  Fill(out,0.0f);
  if(M < TILE || P < TILE || N < TILE){
  // Threads per block: BASE_THREAD_NUM = 256
  // Blocks in each dimension: ceil( (float) M*P / BASE_THREAD_NUM)
    CudaDims dim = CudaOneDim(M*P);
    MatmulKernel_naive<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  }else{
    dim3 block(TILE, TILE);
    dim3 grid((M - 1) / TILE + 1, (P - 1) / TILE + 1);
    MatmulKernel_tiled<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  }
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, const size_t reduce_size, const size_t size){
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) return;
    scalar_t max_val = a[gid * reduce_size];
    for (size_t i = gid * reduce_size + 1; i < (gid+1)*reduce_size; ++i ){
        max_val = fmaxf(max_val, a[i]);
    }
    out[gid] = max_val;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  INIT_DIM;
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END YOUR SOLUTION
}


__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, const size_t reduce_size, const size_t size){
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= size) return;
    scalar_t sum_val = a[gid * reduce_size];
    for (size_t i = gid * reduce_size + 1; i < (gid+1)*reduce_size; ++i ){
        sum_val += a[i];
    }
    out[gid] = sum_val;
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
    INIT_DIM;
    ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END YOUR SOLUTION
}

__global__ void StackKernel(scalar_t **arr, size_t size, size_t total_size,
                            scalar_t *out) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid  < total_size){
        size_t no = gid / size;
        size_t offset = gid % size;
        out[gid] = arr[no][offset];
    }
}

void Stack(const std::vector<CudaArray *> arr, size_t size, CudaArray* out){
//    size is the product of arr[0].shape
    CudaDims dim = CudaOneDim(out->size);
    size_t n = arr.size();
    scalar_t **host_ptr = (scalar_t **)std::malloc(n * sizeof(arr[0]->ptr));
    if (host_ptr == 0)
        throw std::bad_alloc();
    for (size_t i = 0; i < n; ++i ){
        host_ptr[i] = arr[i]->ptr;
    }
    scalar_t **arr_ptr = nullptr;
    cudaError_t error = cudaMalloc(&arr_ptr, n * sizeof(arr[0] -> ptr));
    if (error!= cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
    error = cudaMemcpy(arr_ptr, host_ptr, n*sizeof(arr[0]-> ptr),
                        cudaMemcpyHostToDevice);
    if (error!= cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));

    StackKernel<<<dim.grid, dim.block>>>(arr_ptr, size, out->size, out->ptr);
}

__global__ void SplitKernel(const scalar_t *A, size_t size, size_t total_size,
                            scalar_t **out) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < total_size) {
    int no = gid / size;
    int offset = gid % size;
    out[no][offset] = A[gid];
  }
}

void Split(const CudaArray &A, uint32_t size, std::vector<CudaArray *> out) {
  CudaDims dim = CudaOneDim(A.size);
  size_t n = out.size();

  // copy array of pointers to device
  scalar_t **host_ptr = (scalar_t **)std::malloc(n * sizeof(out[0]->ptr));
  if (host_ptr == 0)
    throw std::bad_alloc();
  for (int i = 0; i < n; ++i) {
    host_ptr[i] = out[i]->ptr;
  }

  scalar_t **arr_ptr = nullptr;
  cudaError_t error = cudaMalloc(&arr_ptr, n * sizeof(out[0]->ptr));
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
  error = cudaMemcpy(arr_ptr, host_ptr, n * sizeof(out[0]->ptr),
                     cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));

  SplitKernel<<<dim.grid, dim.block>>>(A.ptr, size, A.size, arr_ptr);
}

}  // namespace cuda
}  // namespace needle


PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);
  m.def("ewise_sin", EwiseSin);
  m.def("ewise_cos", EwiseCos);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  m.def("stack", Stack);
  m.def("split", Split);
}