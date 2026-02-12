import numpy
import os


graph = numpy.load("./adjacency_matrix.npy")
mx = 0
mi = 100
for row in range(len(graph)):
    cnt = 0
    for col in range(len(graph[0])):
        if graph[row][col] == 1:
            cnt += 1
    if cnt > mx:
        mx = cnt
    if cnt < mi:
        mi = cnt


print(mx, mi)


print(graph.sum(0))