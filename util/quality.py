'''Various measures of clustering quality.

Author: Dan Marchese
'''
import math

def entropy(clustering):
    '''Calculates the entropy of a given clustering.'''

    # Calculate the total number of points in the clustering.
    total_points = float(reduce(lambda x, y: x + sum(y.values()), clustering, 0))

    # Used to accumulate the entropy values for each cluster.
    ent = 0.0

    # Calculate the entropy of each cluster.
    for cluster in clustering:
        sub_ent = 0.0

        # Find the number of points in this cluster.
        num_points = float(sum(cluster.values()))

        # Calculate the entropy for each label.
        if len(cluster) > 1:
            for label in cluster.keys():
                    p = float(cluster[label]) / num_points
                    sub_ent += p * math.log(1.0 / p, len(cluster))

        # Put this cluster into the total entropy accumulator
        ent += (num_points / total_points) * sub_ent

    return ent