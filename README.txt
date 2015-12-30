Lab Assignment 3
Daniel Marchese <marchese.29@osu.edu>
Peter Jacobs <jacobs.269@osu.edu>
CSE 5243: Introduction to Data Mining - Dr. Srinivasan Parthasarathy


SUBMISSION CONTENTS
The submission contains all of the files from the project.  It contains at its root the driver
program for both of the clustering algorithms, a utility function for calculating entropy, and
several data files that were used for testing.


PROJECT STRUCTURE
The project is structured as follows
/
├── README.txt
├── clustering
│   ├── __init__.py
│   ├── dbscan.py
│   └── kmeans.py
├── data
│   ├── document_matrix_1000.pickle
│   ├── document_matrix_256.pickle
│   ├── document_matrix_256_binary_np.pickle
│   ├── document_matrix_256_cont_np.pickle
│   └── reducedFeatureVectorMatrix1000.pkl
├── dbscandriver.py
├── kmeansdriver.py
├── report.pdf
├── results
│   ├── cosine1000.pkl
│   └── euclidean1000.pkl
└── util
    ├── __init__.py
    └── quality.py

Important files are outline below:
dbscandriver.py
Contains all of the logic for executing the dbscan algorithm.

kmeansdriver.py
Contains all of the logic for executing the kmeans algorithm.

clustering/dbscan.py
Contains the actual logic of the dbscan implementation.

clustering/kmeans.py
Contains the actual logic of the kmeans implementation.

util/quality.py
contains code for testing the entropy of a given clustering.

results/cosine1000.pkl and results/euclidean1000.pkl
These files contain the results from the test-suite on K-means.  They were used for generating the
graphs that are found in the report.


RUNNING THE CODE
The code contains 3 external dependencies.  You will need to install numpy, scipy, and matplotlib.
The matplotlib code creates graphs, so you will need to have an X session running to see the
visualizations. (It may crash if no session is available).


RUNNING DBSCAN
The DBSCAN algorithm can be run with the following command:
> python dbscandriver.py


RUNNING K-MEANS
The entire test-suite is built into the kmeans driver.  The only point of configuration is the range
of K values you would like to test, which can be found in the kmeansdriver.py file.
> python kmeansdriver.py


OWNERSHIP
All code used in DBSCAN was written by Peter Jacobs.
All K-Means code as well as the entropy code was written by Dan Marchese


CREDITS
The following open-source libraries are leveraged by this project.
 - numpy - http://www.numpy.org/
    - Used for fast euclidean distance between feature vectors.
 - scipy - http://docs.scipy.org/doc/
    - Used for fast cosine distance between feature vectors.
 - matplotlib - http://matplotlib.org/
    - Used for data visualizations in the K-Means test suite.
