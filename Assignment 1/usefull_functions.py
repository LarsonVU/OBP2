import math
import random

def matmul(m1, m2):
    return [[sum(a*b for a,b in zip(col, row)) for col in zip(*m2)] for row in m1]

def sample_exponential(rate):
    return -math.log(1.0 - random.random()) / rate

def find_min_and_index(arr):
    if not arr:
        return
    
    min = float('inf')
    min_index = -1

    for index, element in enumerate(arr):
        if element < min:
            min = element
            min_index = index
    
    return min, min_index

def transpose(m):
    if not m:
        return []
    
    rows, cols = len(m), len(m[0])
    t = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            t[j][i] = m[i][j]

    return t

matrix = [
    [1, 2, 3],
    [4, 5, 6]
]