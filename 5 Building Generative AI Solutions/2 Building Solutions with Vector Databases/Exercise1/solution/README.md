# Basic Vector Search Implementation

For this exercise we will implement basic vector search
from scratch with just numpy. This will give us a feel
for what's happening under the hood in vector databases.

## Euclidean distance

There are many ways to measure the distance between two vectors.
Let's write a function that computes the `Euclidean distance` 
between vectors. 

This function should take as input two vectors and return
the euclidean distance between them.

For more details you can read this [kaggle page](https://www.kaggle.com/code/paulrohan2020/euclidean-distance-and-normalizing-a-vector)

## KNN search

Using the distance function you just wrote, write a function that 
finds the k-nearest neighbors of a query vector.

This function should take as input a query vector, a 2d array of database vectors,
and an integer k the number of nearest neighbors to return. And it should return 
the vectors that are the k-nearest neighbors of the query vector.

## Other distance metrics

For this problem we'll write a new distance function and modify 
our nearest neighbors function to accept a distance metric.

1. Write a function that computes the [cosine distance](ttps://en.wikipedia.org/wiki/Cosine_similarity) between vectors.
2. Rewrite the `find_nearest_neighbors` function to accept a distance metric.

### Hint:

Please make sure you understand the difference between cosine similarity and cosine distance

## Exploration

Now that we have a nearest neighbors function that accepts a distance metric,
let's explore the differences between Euclidean distance and cosine distance.
Would you expect the same or different answers?

