# Notes
Parallel Programming Notes
-Multiprogramming: multiprocess in one processor
-MultiProcessing: multi process in mul processor
-distributed: multiple process across multi systems

concurrency
-can be simulated but true requires hardware

java thread states:
  -running
  -ready to run
  
  -suspended
  -resumed
  
  -blocked
  -terminated
Thread: 
  -implement runnable
  -extend thread
  
Design Issues:
  -communication between processes
  -sharing/competing
  -synchronization
  -scheduling
potential issues:
  -mutual exclusion
      two processes require non shareable resource. "critical resource"/"critical section"
  -deadlock
      circular depency that blocks both processes.
        should not use destroy() resume() stop() suspend()
  -starvation
  -coherence
  
  
"volatile" keyword

java.util.concurrent
Synchronizers:
  - Semaphore: classic counting semaphore
      > acquireUninterruptibily() -> release()
      > Semaphore(value, boolean fair)
  - ReentrantLock: reentrant mutex
      > lock() by n times, must unlock() by n times
  - ReentrantReadWriteLock
  - CountDownLatch
      > will wait until number of threads desired 
      new CountDownLatch(n)
  - Exchanger
  - Exchanger<T> buff = new Exchanger<>()
  - Executor
  -   java.util.concurrent.ExecutorService /Executors
      > ExecutorService service = Executers.newFixedThreadPool()
      > 
  must implement:::Callable<T> a thread that returns values. 
      > call() function
  Future<T> submit(Callable<T> task)
  get()
Fork/Join daemon threads -> ends when main ends
  ForkJoinTask
  ForkJoinPool
    RecursiveAction
       compute() similar to run/call
        new ForkJoinPool().invoke() -> recursion. invokeAll(blarecuract,blarecuract)
        @Override protected void compute()
    RecursiveTask
Fork/Join suited for divide and conquer strategy
  
Stream api:
  
  Does Not alter data source
  
  Data Source -> Optional Intermediate operation to produce new stream -> Terminal operation to produce result.
  Lazy Evaluation. allows delay in inter operations until terminal oper requires em.
  Terminal Operations
    - collect
    - count
    - forEach
    - max
    - reduce(0, (subtotal, element) -> subtotal + element)
  Intermediate Operations
    - parallel
      - associative, noninterfering, stateless
    - sequential
    - filter(item -> conditionOnItem)
    - mapToInt
    - map()
    - sorted
  
M[Row][p*TILE_WIDTH+tx]
M[Row*Width + p*TILE_WIDTH + tx]
  
#Matrix Multiplier
__global__ void MatrixMulKernel(float* M, float* N, float* P, Int Width)
{
  __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  float Pvalue = 0;
  // Loop over the M and N tiles required to compute the P element
  for (int p = 0; p < n/TILE_WIDTH; ++p) {
    // Collaborative loading of M and N tiles into shared memory
    ds_M[ty][tx] = M[Row*Width + p*TILE_WIDTH+tx];
    ds_N[ty][tx] = N[(t*TILE_WIDTH+ty)*Width + Col];
    __syncthreads();
  }
  for (int i = 0; i < TILE_WIDTH; ++i)Pvalue += ds_M[ty][i] * ds_N[i][tx];
  __synchthreads();
  }
  P[Row*Width+Col] = Pvalue;
}
#Cuda Histogram counter
__global__ void histo_kernel(unsigned char *buffer,
long size, unsigned int *histo)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  // stride is total number of threads
  int stride = blockDim.x * gridDim.x;
  // All threads handle blockDim.x * gridDim.x
  // consecutive elements
  while (i < size) {
    int alphabet_position = buffer[i] – “a”;
    if (alphabet_position >= 0 && alpha_position < 26)
    atomicAdd(&(histo[alphabet_position/4]), 1);
    i += stride;
  }
}
                                                      
#Privatization
__global__ void histo_kernel(unsigned char *buffer,
long size, unsigned int *histo)
{
  __shared__ unsigned int histo_private[7];
  if (threadIdx.x < 7) histo_private[threadidx.x] = 0;
  __syncthreads();
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  // stride is total number of threads
  int stride = blockDim.x * gridDim.x;
  while (i < size) {
    atomicAdd( &(private_histo[buffer[i]/4), 1);
    i += stride;
  }
}
#1D Convolution
__global__ void convolution_1D_basic_kernel(float *N, float *M,
float *P, int Mask_Width, int Width)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  float Pvalue = 0;
  int N_start_point = i – (Mask_Width/2);
  for (int j = 0; j < Mask_Width; j++) {
    if (N_start_point + j >= 0 && N_start_point + j < Width) {
      Pvalue += N[N_start_point + j]*M[j];
    }
  }
  P[i] = Pvalue;
}
__global__ void convolution_2D_kernel(float *P,
  float *N, height, width, channels,
  const float __restrict__ *M) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row_o = blockIdx.y*O_TILE_WIDTH + ty;
  int col_o = blockIdx.x*O_TILE_WIDTH + tx;
  int row_i = row_o - 2;
  int col_i = col_o - 2; 
  if((row_i >= 0) && (row_i < height) &&
  (col_i >= 0) && (col_i < width)) {
    Ns[ty][tx] = data[row_i * width + col_i];
  } else{
    Ns[ty][tx] = 0.0f;
  }
  float output = 0.0f;
  if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH){
    for(i = 0; i < MASK_WIDTH; i++) {
      for(j = 0; j < MASK_WIDTH; j++) {
        output += M[i][j] * Ns[i+ty][j+tx];
    }
  }
 if(row_o < height && col_o < width)
  data[row_o*width + col_o] = output;
}
                                    
#Reduction 1D
The reduction ratio is:
MASK_WIDTH * (O_TILE_WIDTH)/(O_TILE_WIDTH+MASK_WIDTH-1)
#Reduction 2d
O_TILE_WIDTH^2 * MASK_WIDTH^2 /(O_TILE_WIDTH+MASK_WIDTH-1)^2

#Parallel Sum Reduction
__shared__ float partialSum[2*BLOCK_SIZE];
unsigned int t = threadIdx.x;
unsigned int start = 2*blockIdx.x*blockDim.x;
partialSum[t] = input[start + t];
partialSum[blockDim+t] = input[start + blockDim.x+t];
for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
{
  __syncthreads();
  if (t < stride)
  partialSum[t] += partialSum[t+stride];
}
#Work ineficient scan kernel
__global__ void work_inefficient_scan_kernel(float *X, float *Y, int InputSize) {
  __shared__ float XY[SECTION_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < InputSize) {XY[threadIdx.x] = X[i];}
  // the code below performs iterative scan on XY
  for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
  __syncthreads();
    float in1 = XY[threadIdx.x - stride];
  __syncthreads();
    XY[threadIdx.x] += in1;
  }
  __ syncthreads();
  If (i < InputSize) {Y[i] = XY[threadIdx.x];}
}
#cudaStream
  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  float *d_A0, *d_B0, *d_C0; // device memory for stream 0
  float *d_A1, *d_B1, *d_C1; // device memory for stream 1
  // cudaMalloc() calls for d_A0, d_B0, d_C0, d_A1, d_B1, d_C1 go
  here
#cudaStreamExample
  cudaMemcpyAsync(d_A0, h_A+i, SegSize*sizeof(float),…, stream0);
  cudaMemcpyAsync(d_B0, h_B+i, SegSize*sizeof(float),…, stream0);
  cudaMemcpyAsync(d_A1, h_A+i+SegSize, SegSize*sizeof(float),…, stream1);
  cudaMemcpyAsync(d_B1, h_B+i+SegSize, SegSize*sizeof(float),…, stream1);
  vecAdd<<<SegSize/256, 256, 0, stream0>>>(d_A0, d_B0, …);
  vecAdd<<<SegSize/256, 256, 0, stream1>>>(d_A1, d_B1, …);
  cudaMemcpyAsync(h_C+i, d_C0, SegSize*sizeof(float),…, stream0);
  cudaMemcpyAsync(h_C+i+SegSize, d_C1, SegSize*sizeof(float),…, stream1);
  
  cudaStreamSynchronize(stream_id)
#more cuda examples
  int *hostArray;
  int *gpuArray;
  int threads = 256;
  int blocks = ceil(N/threads);

  #ifdef PAGED
  hostArray = (int*) malloc(N*sizeof(int));
  #else
  cudaHostAlloc((void**)&hostArray, N*sizeof(int), cudaHostAllocDefault );
  #endif

  cudaMalloc(&gpuArray, N*sizeof(int));

  cudaMemcpy(gpuArray, hostArray, N*sizeof(int), cudaMemcpyHostToDevice);
  doubleVector<<<blocks,threads>>>(gpuArray,N);

  cudaFree(gpuArray);
  #ifdef PAGED
  free(hostArray);
  #else
  cudaFreeHost(hostArray);
  #endif

  cudaDeviceSynchronize();
  return 0;
  
#Additional Notes
  SPMD is by far the most commonly used pattern for
structuring massively parallel programs.
-good: Accelerate legacy codes
-better: rewrite create new codes
-best: rethink numerical methods/algorithms
 Reduction: Summarize a set of input values into one value using a
“reduction operation”
