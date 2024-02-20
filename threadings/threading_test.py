from threading import Thread, Lock
import time 
from queue import Queue
database_value =0
def increase(lock):
    global database_value
    lock.acquire()
    # with lock:
    local_cp =database_value
    #processing
    local_cp+=1
    time.sleep(0.1)
    database_value = local_cp
    lock.release()


if __name__ == "__main__" :
    print("start : ",database_value)
    lock = Lock()
    th1 = Thread(target=increase,args=(lock,))
    th2 = Thread(target=increase,args=(lock,))
    th1.start()
    th2.start()

    th1.join()
    th2.join()

    print("end value ", database_value)
    print("end main")


