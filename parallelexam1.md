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
