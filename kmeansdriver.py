'''This is where the logic for driving the k-means algorithm sits.

Author: Dan Marchese
'''
from collections import OrderedDict
import cPickle
import os
import sys
import time

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
    xs = range(2, 41)
    es = []  # Entropy values
    vs = []  # Variance values
    ts = []  # Elapsed time values
    for k in xs:
        print 'Running %d-Means' % k
        start = time.time()
        clustering, iterations = run_kmeans(k, vectors, distance_metric)

        # Calculate the statistics we are graphing
        ts.append((time.time() - start) / iterations)
        e_clustering = get_entropy_clustering(clustering)
        es.append(entropy(e_clustering))
        mean = float(reduce(lambda x, y: x + sum(y.values()), e_clustering, 0.0)) / len(clustering)
        var = float(sum(map(lambda x: (sum(x.values()) - mean)**2, e_clustering))) / len(clustering)
        vs.append(var)
    return (xs, es, vs, ts)

def run_test_suite(vectors):
    '''Runs the suite of tests on the given vectors and graphs the results.'''

    test_start = time.time()

    # Test suite using euclidean distance.
    print 'Running tests using euclidean distance.'
    xs, es, vs, ts = run_tests_with_metric(vectors, euclidean_distance)
    with open('results/euclidean1000.pkl', 'w') as f:
        cPickle.dump((xs, es, vs, ts), f)

    # Graph the entropy results
    ax = plt.subplot(321)
    plt.title('K-Means Entropy - Euclidean Distance')
    plt.ylabel('Entropy')
    plt.plot(xs, es, 'bo-')

    # Graph the variance results
    ax = plt.subplot(323)
    plt.title('K-Means Cluster Variance - Euclidean Distance')
    plt.ylabel('Variance')
    plt.plot(xs, vs, 'bo-')

    # Graph the execution time
    ax = plt.subplot(325)
    plt.title('K-Means Runtime Per Iteration - Euclidean Distance')
    plt.ylabel('Time (s)')
    plt.plot(xs, ts, 'bo-')

    # Test suite using cosine distance.
    print 'Running tests using cosine distance.'
    xs, es, vs, ts = run_tests_with_metric(vectors, cosine_distance)
    with open('results/cosine1000.pkl', 'w') as f:
        cPickle.dump((xs, es, vs, ts), f)

    # Graph the entropy results
    ax = plt.subplot(322)
    plt.title('K-Means Entropy - Cosine Distance')
    plt.ylabel('Entropy')
    plt.plot(xs, es, 'bo-')

    # Graph the variance results
    ax = plt.subplot(324)
    plt.title('K-Means Cluster Variance - Cosine Distance')
    plt.ylabel('Variance')
    plt.plot(xs, vs, 'bo-')

    # Graph the execution time
    ax = plt.subplot(326)
    plt.title('K-Means Runtime Per Iteration - Cosine Distance')
    plt.ylabel('Time (s)')
    plt.plot(xs, ts, 'bo-')

    test_end = time.time()

    #plt.tight_layout()
    plt.savefig('results/plots.png')
    plt.show()
    return test_end - test_start

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
    elapsed = run_test_suite(vectors)
    print 'Elapsed Time: %f hours' % (elapsed / 60.0 / 60.0)

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('\nReceived keyboard interrupt...aborting.')
    except Exception as e:
        sys.exit(str(e))
