from multiprocessing import Process, Value ,Array,Lock
import os
import time

def square_numbers():
    for i in range(1000):
        i * i
def add_100(number,lock):
    for i in range(100):
        time.sleep(0.01)
        """
        lock.acquire()
        number.value+=1
        lock.release()
        #or
        """
        with lock:
            number.value+=1    
def add_100_arr(array,lock):
    for _ in range(100):
        time.sleep(0.01)
        for i in range(len(array)):
            with lock:
                array[i]+=1


if __name__ == "__main__":
    processes = []
    lock = Lock()
    num_processes = os.cpu_count()

    shared_number = Value('i' , 0)
    shared_array = Array("d", [0.0,12.3,456.7])

    print("shared variable : ",shared_array[:])

    p1 = Process(target=add_100_arr,args=(shared_array,lock))
    p2 = Process(target=add_100_arr,args=(shared_array,lock))

    p1.start()
    p2.start()
    # race accurs if you don't lock!
    p1.join()
    p2.join()
    print("number at end is ", shared_array[:])
    # tutorial that start and join processes
    """
    for i in range(num_processes):
        process = Process(target=square_numbers)
        processes.append(process)
    for process in processes:
        process.start()

    for process in processes:
        process.join() 
    """