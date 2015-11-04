import random

import numpy as np

def is_same(part1, part2):
    '''Determines whether the given partitionings of the data are equivalent.'''
    assert part1.keys() == part2.keys()
    for num, points in part1.iteritems():
        p2points = part2[num]
        if len(p2points) != len(points):
            return False
        elif set([point[0] for point in p2points]) != set([point[0] for point in points]):
            return False
    return True

def _generate_partitions(data, centers, distance):
    partitioning = { i: [] for i in centers.keys() }
    for vector in data:
        current_min_dist = 1000000.0
        current_min = None
        for cluster_num in centers.keys():
            dist = distance(centers[cluster_num], vector[2])
            if dist < current_min_dist:
                current_min_dist = dist
                current_min = cluster_num
        partitioning[current_min].append(vector)
    return partitioning

def _recalculate_centers(partitioning, rand_vec):
    centers = dict()
    for partition, vectors in partitioning.iteritems():
        if len(vectors) > 0:
            # Calculate the new centroid
            centers[partition] = reduce(lambda x, y: x + y[2], vectors, np.zeros(1000))
            centers[partition] /= len(vectors)
        else:
            # Re-initialize this centroid since it doesn't have any attached vectors.
            centers[partition] = rand_vec[2].astype(float)

    return centers

def run_kmeans(num_centroids, data, distance):
    '''Computes the K-Means clustering on the given data using the given distance function.'''
    # Randomly initialize the cluster centers.
    centers = { i: np.random.rand(1000) for i in range(num_centroids) }

    # Start the partitioning.
    print '\tIteration 1'
    new_partitioning = _generate_partitions(data, centers, distance)
    # Calculate the initial centroids
    centers = _recalculate_centers(new_partitioning, data[random.randint(0, len(data)-1)])

    i = 2
    old_partitioning = { i: [] for i in centers.keys() }
    while not is_same(old_partitioning, new_partitioning):
        print '\tIteration %d' % i
        i += 1
        # Store the old partitioning for later comparison
        old_partitioning = new_partitioning.copy()

        # Re-partition the data and calculate the new centers.
        new_partitioning = _generate_partitions(data, centers, distance)
        centers = _recalculate_centers(new_partitioning, data[random.randint(0, len(data)-1)])

    # A mapping from center location -> points in cluster around the centroid.
    return [points for num, points in new_partitioning.iteritems()]
