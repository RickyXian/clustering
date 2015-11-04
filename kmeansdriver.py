'''This is where the logic for driving the k-means algorithm sits.

Author: Dan Marchese
'''
from collections import OrderedDict
import cPickle
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine as cos_distance

from clustering.kmeans import run_kmeans
from util.quality import entropy

def euclidean_distance(x, y):
    '''Calculates the euclidean distance between x and y.'''
    return np.linalg.norm(x - y)

def cosine_distance(x, y):
    '''Calculates the cosine distance between x and y.'''
    return cos_distance(x, y)

def vector_from_dictionary(dictionary):
    '''Converts the given dictionary to a list of numbers.'''
    od = OrderedDict(sorted(dictionary.items()))
    return [value for key, value in od.iteritems()]

def get_entropy_clustering(clustering):
    result = []
    for vectors in clustering:
        current = dict()
        for vec in vectors:
            if vec[1] not in current:
                current[vec[1]] = 1
            else:
                current[vec[1]] += 1
        result.append(current)
    return result

def run_tests_with_metric(vectors, distance_metric):
    '''Runs the suite of tests using the given distance metric.'''
    xs = range(2, 40)
    ys = []
    for k in range(2, 40):
        print 'Running %d-Means' % k
        clustering = run_kmeans(k, vectors, distance_metric)
        ys.append(entropy(get_entropy_clustering(clustering)))
    return (xs, ys)

def run_test_suite(vectors):
    '''Runs the suite of tests on the given vectors.'''

    # Test suite using euclidean distance.
    print 'Running tests using euclidean distance.'
    xs, ys = run_tests_with_metric(vectors, euclidean_distance)
    ax = plt.subplot(211)
    plt.title('K-Means Entropy using Euclidean Distance')
    plt.xlabel('K')
    plt.ylabel('Entropy')
    plt.bar(xs, ys)

    # Test suite using cosine distance.
    print 'Running tests using cosine distance.'
    xs, ys = run_tests_with_metric(vectors, cosine_distance)
    plt.subplot(212)
    plt.title('K-Means Entropy using Cosine Distance')
    plt.xlabel('K')
    plt.ylabel('Entropy')
    plt.bar(xs, ys)

    plt.tight_layout()
    plt.show()

def main():
    '''The main entry point for the program.'''

    # Load the feature vectors
    print 'Loading the feature vectors.'
    with open('data/document_matrix_1000.pickle', 'r') as f:
        data = cPickle.load(f)

    # Process the incoming vectors
    print 'Indexing the feature vectors.'
    vectors = []
    for vector in [v for v in data if 'NULL' not in v[1]]:
        np_vector = np.array(vector_from_dictionary(vector[3]), dtype=int)
        # Use this opportunity to remove null vectors
        if np_vector.any():
            for clazz in vector[1]:
                vectors.append([vector[0], clazz, np_vector])

    # Run the tests suite.
    run_test_suite(vectors)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        sys.exit(str(e))
