# Project-plutus documentation is in ProjectDetails.pdf

## Docstring Information

LearningModel is class that is used to make a model that
Continually builds and expands on neural nets in an attempt to build
a neural net to interpret the movement of the solution space from the
data provided

Methods Employed:
The neural networks are basic feedforward artificial neural networks. The neural network manipulation algorithm is based off of a paper written by Kenneth O. Stanley and Risto Miikkulainen by the name of NeuroEvolution of Augmenting Topologies (NEAT). This system is interesting as it is possible to 'optimize and complexify solutions simultaneously. Inspiration for this comes from SethBling and his MarI/O.

Documentation Points:
1.  The file format for saving and restoring generations is described below. The format is JSON inspired with implicit attributes based on the structure of the developed classes. This sacrifices readability in order to decrease file size by a factor of at least 2:1 (less conservative estimates seems closer to a factor 4:1).

Files will begin with '{' and end with '}' and will encompass exactly one pool. A default pool will look as follows '{[],0,0,0,0,0}'

A pool with one species with one genome with one default gene will look as follows '{[{0,0,[{[{0,0,0,F,0}],0,0,0,0,0,0,0;0;0;0;0;0;0}],0}],0,0,0,0,0}'

One important note is that the network attribute of the Genome class will always be absent from this file. Storing the network class would be redundant and we currently value efficient memory usage over redundancy. The purpose of this file structure is decreased readability, in order to conserve the limited memory we have available to us. Files of this type will be marked .ion for Implicit Object Notation.

2.  Two adjacency lists were used to represent the graph. For one adjacency list at index i there were a list of elements such that any element j in that list represents an edge (i,j) in the graph. In the second adjacency list at index i there was a list of elements such that any element j in that list represents an edge (j,i) in the graph. In comparison to a single adjacency list, this has the advantage of improving the average case run time. For a single adjacency list the best, average, and worse case search time for either incoming or outgoing nodes (depending on which direction on the adjacency lists' implementation) is O(E). For a double adjacency list the best case search time is O(1) and the average case search time depends on the sparsity of the directed acyclic graph, but can be no worse O(E). Having twice the adjacency lists comes with the obvious side-effect of double the space complexity. Using two adjacency lists has the advantage of saving space, as an adjacency matrix costs O(V^2) whereas an adjacency list costs O(V*E). On top of that searching for all the edges off of a node takes best, average, and worst case O(E), whereas the double adjacency list does better than that (as discussed before)

3.  I tried to find literature about the most meaningful way to select individuals to engage in crossover given speciation information, but was unable to find anything. After consideration I have chosen to allow crossover between any two individuals in the population. The reason is this, forcing crossover between individuals of the same species would mean finding the local optimum that the species is converging too more efficiently, but we will likely miss out on any optimum that are mixtures of functionality across species. Specifically, forcing crossover intra-species means will force converging to a local optimum faster, but since the "Pareto Front" we maintain is actually a subset of the Pareto Front we are more likely to miss out on the global optimum.

Ideas Used:
1.  NEAT (obvious)
2.  Kahn's Topological Sort (Topological ordering of genes)
3.  Fitness Proportionate Selection
4.  Implicit Object Notation (File storage)
5.  Niching or Speciation which is inherent in NEAT

author: Andrew Butler
