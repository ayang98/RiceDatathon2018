import numpy as np
import heapq as hq
from collections import defaultdict

def knn(training, test, k, output):
    """
    training = training data in the form of a numpy matrix where each row represents an entry of data
    test = numpy matrix of data to be tested for outcome knn
    k = k nearest neighbors
    output = corresponding numpy outcome array to each row of training data
    """
    predicted = [] #the matrix holding the predicted outcomes using knn
    for array1 in test:
        outcomes = defaultdict(int)
        distances = []
        max_value = 0
        for array2 in training:
            hq.heappush(distances, (numpy.linalg.norm(array2[1:]-array1), array2))
        for dummy_index in range(k):
            array = hq.heappop(distances)[1]
            outcomes[output[array[0]] += 1
        for key, value in outcomes.items():
            if value > max_value:
                max_value = value
                max_key = key
        predicted.append(max_key)
    return np.transpose(np.array([predicted]))
