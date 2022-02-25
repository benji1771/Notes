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
