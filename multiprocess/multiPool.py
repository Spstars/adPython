from multiprocessing import Pool 
import time
import os
def cube(number):
    time.sleep(0.01)   
    print(f"pid : {os.getpid()}, number : {number}")
    return number * number * number


if __name__ == "__main__":
    numbers = range(30)
    p = Pool(1)
    start_time = time.time()
    result = p.map(cube,  numbers)
    p.close()
    p.join()
    end_time = time.time()
    print(result[3],end_time-start_time)

