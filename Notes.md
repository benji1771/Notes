

#GPU-accelerated Vs. CPU-only Application
    cudaMallocManaged(); //unified memmory
    cudaDeviceSynchronize(); //synchronize gpu with cpu

    try to initialize data in kernel.

    cudaMemPrefetchAsync(data, size, deviceID/cudaCpuDeviceID)



#CUDA Programming

    ##Streams
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    functName<<<1,1,0,stream>>>();
    cudaStreamDestroy(stream);


    __global__ void name(){} Kernel function must be void
    name<<<blocks,threads>>>(); invoke/running kernel.

    
    gridDim.x -> x dimension of grid
    blockDim.x -> x dimension of block
    blockIdx.x -> x block index in grid
    threadIdx.x -> x threada index in grid
    blockDim.x * blockIdx.x + threadIdx.x = thread index calculation

    ##Grids and Blocks dimensions

    dim3 threads_per_block(16, 16, 1) sets blockdimX
    dim3 number_of_blocks(16, 16, 1)
    
    ##when data too large
    indexWithinGrid = threadIdx.x + blockIdx.x * blockDim.x
    gridStride = gridDim.x * blockDim.x
    for(indexWithinGrid to N; in gridStride increment){
        work on index
    }




    edge cases make sure index less than allowed working size.
    Number of blocks = (SizeOfData + NumberOfThreads - 1) / NumberOfThreads | Must pad for edge case with one extra block.

    Blocks of threads scheduled on SMs
    Blocks divisible by number of SMs
    threads divisible by number of 32


    int deviceId;
    cudaGetDevice(&Id)
    cudaDeviceProp props;
    cudaGetDeviceProperties(&prop, deviceId)

    computeCapabilityMajor = props.major
    int computeCapabilityMinor = props.minor;

    ##Profiling
    add -f to overwrite report file
    nsys profile --states=true ./the-output-file



GPU:
    Grid
        Blocks 
            Threads (Each Block must have same number of threads)
        
    Grid Dimension ->  Number of blocks
    Block Dimension -> Number of Threads per block
    Thread


#Compilation
    nvcc
    -arch=sm_70 <-> streaming architecture
