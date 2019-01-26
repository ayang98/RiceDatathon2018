import numpy

x = numpy.array([[1,1,1,1,1],[2,2,2,2,2]])
print x

numpy.stack((x,numpy.array([[1],[0]])),axis = -1)
print x
