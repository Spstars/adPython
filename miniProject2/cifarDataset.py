#data.DataLoader(dataset=tr_dataset, batch_size=128, num_workers=8, shuffle=True)
#making batches, shuffle, num_workers
#using generator
#multiprocessing
#need file system to load images

from multiprocessing import Pool
import numpy 
import time
import sys
def measure_time(func):
    def decorating(*args, **kwargs):
        start = time.time
        result = func(*args, **kwargs)
        end = time.time
        print("spend on loading datasets : ", end-start)
        return result
    return decorating

def download_cifar10():
    print("downloading...")

def cifarDataLoader(batch_size = 64 ,num_workers =2 ,shuffle = True):
    yield 1