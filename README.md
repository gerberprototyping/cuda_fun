# Fun with CUDA C++

I have been experimenting with matrix multiplication on both x86 and Nvidia GPU
using CUDA C++. Multiple algorithms have been tested with multiple matrix sizes
and multiple data types. Code is located in [`matmul.cu`](./matmul/matmul.cu)

The GPU algorithms have not yet been optimized very much.

### Algorithms used
- `gpu_matmul()`: single for-loop per thread
- `gpu_matmul_trans()`: transposed right-side matrix, single for-loop per thread
- `cpu_matmul()`: simple triple for-loop (all CPU implementation are single-thread)
- `cpu_matmul_trans()` transposed right-size matrix, triple for-loop
- `avx256_matmul()` utilizes Intel AVX-256 vector instructions (int32 only currently)

### Data Types
- $int32$: 32-bit signed integer
- $int64$: 64-bit signed integer
- $float32$: single-precision 32-bit floating-point
- $float64$: double-precision 64-bit floating-point

### Matrix Sizes
- $16$ MiB
- $32$ MiB
- $64$ MiB $~~$*in-progress...*
- $128$ MiB $~$*in-progress...*
- $256$ MiB $~$*maybe?*


## Results

### Results: 16 MiB / 32 MiB

The 4 MiB and 8 MiB results are grouped together because they result in the same
number of total operations for 32-bit and 64-bit datatypes respectively.

*Unless noted, a 16 MiB matrix size was used*

<br>

**Table 1.** 16MiB / 32MiB results $~~$ *(time in milliseconds)*

| | $int32$ | $int64$ | $int64~$ 32 MiB | $float32$ | $float64$ | $float64~$ 32 MiB
|-|-|-|-|-|-|-|
| **CPU** | $\textcolor{red}{24,290}$ | $2,824$ | $\textcolor{red}{26,546}$ | $\textcolor{red}{25,315}$ | $3,284$ | $\textcolor{red}{26,802}$
| **CPU**$^T$ | $~~3,501$ | $\textcolor{lightgreen}{1,426}$ | $\textcolor{lightgreen}{~~4,121}$ | $\textcolor{yellow}{~~8,136}$ | $2,921$ | $\textcolor{yellow}{~~8,214}$
| **GPU** | $~~3,590$ | $2,098$ | $~~5,946$ | $~~3,590$ | $2,103$ | $~~5,981$
| **GPU**$^T$ | $~~3,536$ | $2,053$ | $~~5,702$ | $~~3,538$ | $2,097$ | $~~5,934$
| **AVX-256** | $\textcolor{lightgreen}{~~1,100}$ | $~~~-$ | $~~~~~~-$ | $~~~~~-$ | $~~~~-$ | $~~~~~-$

There are several very interesting trends that appear in this data:

- The standard CPU seems to favor the the 64-bit datatypes which are a $1,148\times1,448$
  matrix as opposed to the $2,048\times2,048$ matrix of the 32-bit datatypes. This
  is very likely due to cache behaviors such as line size, prefetching, or other.
- The lower $float32$ and $float64$ 32 MiB performance is likely due to relatively
  lower FP performance on the CPU. The $float64$ remains competitive due to the
  cache behavior uplift.
- The nearly order-of-magnitude worse performance of the standard CPU algorithm
  is likely due to cache thrashing or other similar behavior.
- The marginally small gain of the transposed GPU algorithm over the standard
  GPU algorithm demonstrates that the memory system and cache structure must be
  designed to work well with column data accesses patterns as well as more typical
  row accesses patterns. This makes sense as matrix operation (row and column
  accesses) is a core part of GPU workloads.
- Note that GPU performance was measured *after* data had been transferred. The
  calculation time is purely GPU work.


### Results: 64 MiB / 128 MiB

Data gathering in progress...
