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
  - ReentrantLock: reentrant mutex
  - ReentrantReadWriteLock
  - CountDownLatch
