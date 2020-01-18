#include "graph_gen.h"

#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <map>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <numeric>
#include <memory>


namespace graph_gen {

/*********************************************************************  HELPER FUNCTIONS DECL  ********************************************************************/

	void projectValueToRange(double lowerBound, double upperBound, double & value);

	void checkForConnectivity(int currentVertice, int edgesCount, double & nEdges);

	std::vector<int> getRightEndVerticesOfLeftVert(std::multimap<int, int>& graph, int leftVertice, std::vector<int>& rightVertices);

	bool verticesCannotBeConnected(std::multimap<int, int>& graph, int i, int j, std::vector<int>& rightVertices);

	bool tooFewAvailableVerticesOnTheRight(std::multimap<int, int>& graph, int leftVertice, std::vector<int>& rightVertices, int numberOfDesiredEdges);

	unsigned countEdgesFromLeftVertToRightVertices(std::multimap<int, int>& graph, int leftVertice, std::vector<int>& rightVertices);
	
	bool schuffleVectorValid(std::vector<int>& vec, int verticesNumber);

	std::multimap<int, int> addRandomEdgesToBipartite(VecGraph& graph,
		std::vector<bool> & left, std::vector<bool> & right, unsigned desiredNofEdges);

	void generateShufflePattern(std::vector<int>& shufflePattern, int verticesCount);
/*****************************************************************  END OF HELPER FUNCTIONS DECL  ****************************************************************/


/**
* Generate quasi-random tree.
*
* @param verticesCount number of vertices in the tree.
* @param expectedMeanEdgeNumber mean (expected) number of edges each vertex should derive.
* @param standardDeviation standard deviation of the expectedMeanEdgeNumber parameter.
* @return (V, E) graph which is quasi random tree
*/
std::pair<std::vector<int>, std::vector<int>> generateVectorTree(int verticesCount, double expectedMeanEdgeNumber, double standardDeviation) {
	std::vector<int> v, e;
	int maxGeneratedVerticeNumber = 0;

	std::random_device rd;
	std::mt19937 generator(rd());
	// normal distribution with parameters
	std::normal_distribution<double> distribution(expectedMeanEdgeNumber, standardDeviation);

	double nEdges;
	for (int iVert = 0; iVert < verticesCount && maxGeneratedVerticeNumber < verticesCount; ++iVert) {
		// get random number of edges from normal distribution
		nEdges = distribution(generator);
		projectValueToRange(0.0, verticesCount, nEdges);
		// check nEdges is not too small to preserve graph connectivity
		checkForConnectivity(v.size(), maxGeneratedVerticeNumber, nEdges);

		v.push_back(maxGeneratedVerticeNumber);
		for (int j = 0; j < (int)nEdges; ++j) {
			if (maxGeneratedVerticeNumber == verticesCount - 1) {
				break;
			}
			e.push_back(++maxGeneratedVerticeNumber);
		}
	}
	// insert last (additional) element which points to end of e
	v.push_back(e.size());
	return std::make_pair(v, e);
}

/**
* Generate quasi-random tree.
*
* @param verticesCount number of vertices in the tree.
* @param expectedMeanEdgeNumber mean (expected) number of edges each vertex should derive.
* @param standardDeviation standard deviation of the expectedMeanEdgeNumber parameter.
* @return matrix graph which is quasi random tree
*/
int* generateMatrixTree(int verticesCount, double expectedMeanEdgeNumber, double standardDeviation) {
	auto pair = generateVectorTree(verticesCount, expectedMeanEdgeNumber, standardDeviation);
	return convertVectorGraphToMatrixGraph(pair);
}

/**
* Generate quasi-random graph in vector (V, E) representation.
*
* @param verticesCount number of vertices in the graph
* @param approxEdgesCount desired number of edges in the graph
* @param exptdMeanEdgeNumberForTreeVertice mean (expected) number of edges that each
*		 vertex of the graph spanning tree should derive
* @param stdDeviation standard deviation of the expectedMeanEdgeNumber parameter
* @return graph in vector (V, E) representation
*/
VecGraph generateVectorGraph(int verticesCount, int approxEdgesCount, double exptdMeanEdgeNumberForTreeVertice, double stdDeviation) {
	int finalEdgesNumber;
	auto matrixGraph = generateMatrixGraph(verticesCount, approxEdgesCount, exptdMeanEdgeNumberForTreeVertice, stdDeviation, &finalEdgesNumber);

	return convertMatrixGraphToVectorGraph(matrixGraph, verticesCount, finalEdgesNumber);
}

/**
* Generate quasi-random graph in matrix representation.
*
* @param verticesCount number of vertices in the graph
* @param approxEdgesCount desired number of edges in the graph
* @param exptdMeanEdgeNumberForTreeVertice mean (expected) number of edges that each vertex of the graph spanning tree should derive
* @param stdDeviation standard deviation of the expectedMeanEdgeNumber parameter
* @param finalEdgesCount pointer to variable in which final number of edges is stored
* @return graph in matrix representation
*/
int* generateMatrixGraph(int verticesCount, int approxEdgesCount, double exptdMeanEdgeNumberForTreeVertice, double stdDeviation, int* finalEdgesCount) {
	auto tree = generateVectorTree(verticesCount, exptdMeanEdgeNumberForTreeVertice, stdDeviation);
	int* matrixGraph = convertVectorGraphToMatrixGraph(tree);
	*finalEdgesCount = addRandomEdges(matrixGraph, verticesCount, tree.second.size(), approxEdgesCount);

	return matrixGraph;
}

/**
* Generate bipartite graph.
*
* @param verticesCount desired number of vertices
* @param approxEdgesCount approximate number of edges
* @param exptdMeanEdgeNumberForTreeVertice approximate number of edges for every vertice in the spanning tree
* @param stdDeviation standard deviation of exptdMeanEdgeNumberForTreeVertice parameter
* @return bipartite graph given as adjacency vectors (V, E)
*/
VecGraph generateBipartiteVectorGraph(int verticesCount, int approxEdgesCount, double exptdMeanEdgeNumberForTreeVertice, double stdDeviation) {

	auto pair_v_e = generateVectorTree(verticesCount, exptdMeanEdgeNumberForTreeVertice, stdDeviation);
	auto vertices_l_r = bipartiteTree(pair_v_e);
	auto graph = addRandomEdgesToBipartite(pair_v_e,
		vertices_l_r.first, vertices_l_r.second, approxEdgesCount);
	auto graph_v_e = convertMultimapGraphToVectorGraph(graph, verticesCount);

	return graph_v_e;
}

/**
* Generate bipartite graph in matrix representation. Caller function must free memory returned from this function.
*
* @param verticesCount desired number of vertices
* @param approxEdgesCount approximate number of edges
* @param exptdMeanEdgeNumberForTreeVertice approximate number of edges for every vertice in the spanning tree
* @param stdDeviation standard deviation of exptdMeanEdgeNumberForTreeVertice parameter
* @return bipartite graph in matrix representation
*/
int* generateBipartiteMatrixGraph(int verticesCount, int approxEdgesCount, double exptdMeanEdgeNumberForTreeVertice, double stdDeviation) {
	auto pair = generateBipartiteVectorGraph(verticesCount, approxEdgesCount, exptdMeanEdgeNumberForTreeVertice, stdDeviation);
	return convertVectorGraphToMatrixGraph(pair);
}

/**
* Generate graph isomorphic to given input graph
*
* @param graph vector (V, E) representation of the input graph
* @param shufflePattern vector of projection of the input graph vertices (vector indices)
*		 to output graph vertices (vector values)
* @return graph isomorphic to input graph given in vector (V, E) representation
*/
VecGraph generateIsomorphicVectorGraph(VecGraph& graph, std::vector<int> shufflePattern) {
	auto& v = graph.first;
	auto& e = graph.second;
	std::vector<int> new_v(v.size()), new_e(e.size());
	std::vector<int> numberOfEdgesForNewVertices(v.size());

	/* Prepare shuffle pattern vector */
	if (shufflePattern.empty() || !schuffleVectorValid(shufflePattern, v.size() - 1)) {
		generateShufflePattern(shufflePattern, v.size() - 1);
	}

	int verticesNumber = v.size() - 1;

	for (int i = 0; i < verticesNumber; ++i) {
		numberOfEdgesForNewVertices[shufflePattern[i]] = v[i + 1] - v[i];
	}
	int sumOfPlaces = 0;
	for (int i = 0; i < verticesNumber; ++i) {
		new_v[i] = sumOfPlaces;
		sumOfPlaces += numberOfEdgesForNewVertices[i];
	}
	new_v[verticesNumber] = sumOfPlaces;
	for (int i = 0; i < verticesNumber; ++i) {
		int nEdgesForThisVert = v[i + 1] - v[i];
		for (int j = 0; j < nEdgesForThisVert; ++j) {
			new_e[new_v[shufflePattern[i]] + j] = shufflePattern[e[v[i] + j]];
		}
	}

	return std::make_pair(new_v, new_e);
}

/**
* Generate graph isomorphic to given input graph
*
* @param graph matrix representation of the input graph
* @param verticesCount number of vertices in the graph
* @param edgesCount number of edges in the graph
* @param shufflePattern vector of projection of the input graph vertices (vector indices)
*		 to output graph vertices (vector values)
* @return graph isomorphic to input graph given in matrix representation
*/
int* generateIsomorphicMatrixGraph(int* graph, int verticesCount, int edgesCount, std::vector<int> shufflePattern) {
	auto vectorGraph = convertMatrixGraphToVectorGraph(graph, verticesCount, edgesCount);
	auto isomorphicVectorGraph = generateIsomorphicVectorGraph(vectorGraph, shufflePattern);
	return convertVectorGraphToMatrixGraph(isomorphicVectorGraph);
}

/**
* Convert graph given as multimap (where key are vertices and values are edges) to vector graph (V, E).
*
* @param graph multimap graph representation
* @return graph given as adjacency vectors (V, E)
*/
VecGraph convertMultimapGraphToVectorGraph(std::multimap<int, int> & graph, int verticesCount) {
	std::vector<int> v;
	std::vector<int> e(graph.size());

	int currentEdgeIndex = 0;
	for (int i = 0; i < verticesCount; ++i) {
		v.push_back(currentEdgeIndex);
		auto rangeIters = graph.equal_range(i);
		for (; rangeIters.first != rangeIters.second; ++rangeIters.first) {
			e[currentEdgeIndex++] = rangeIters.first->second;
		}
	}
	v.push_back(e.size());
	return std::make_pair(v, e);
};

/**
* Convert graph from matrix representation to vector (V, E) representation.
*
* @param graph matrix representation of the graph
* @param verticesNumber number of vertices in the graph
* @param edgesNumber number of edges in the graph
* @return graph given in vector (V, E) representation
*/
VecGraph convertMatrixGraphToVectorGraph(int* graph, int verticesNumber, int edgesNumber) {
	std::vector<int> v(verticesNumber + 1), e;
	if (edgesNumber <= 0) {
		edgesNumber = std::accumulate(graph, &graph[verticesNumber*verticesNumber], 0);
		if (edgesNumber % 2)
			throw std::exception("Error: graph matrix have eneven number of edges!\n");
		edgesNumber /= 2;
	}
	e.resize(edgesNumber);

	int edgeIndex = 0;
	for (int i = 0; i < verticesNumber; ++i) {
		v[i] = edgeIndex;
		for (int j = i + 1; j < verticesNumber; ++j) {
			if (graph[i * verticesNumber + j] == 1) {
				e[edgeIndex] = j;
				++edgeIndex;
			}
		}
	}
	v[verticesNumber] = e.size();
	return std::make_pair(v, e);
}

/**
* Given graph by vector of vertices and vector of edges this function produces the same graph but using multimap.
* It simplifies inserting new edges to this graph.
*
* @param v representation of graph vertices
* @param e
* @return graph as multimap<int, int> where keys are from v vector and values are from e vector
*/
std::multimap<int, int> convertVectorGraphToMultimapGraph(VecGraph& graph) {
	auto& v = graph.first;
	auto& e = graph.second;
	std::multimap<int, int> multimap_graph;

	for (unsigned i = 0; i < v.size() - 1; ++i) {
		for (int j = v[i]; j < v[i + 1]; ++j)
			multimap_graph.insert({ i, e[j] });
	}
	return multimap_graph;
};

/**
* Convert graph from vector (V, E) representation to matrix representation. 
* Caller function must free memory returned from this function.
*
* @param graph is graph given in vector (V, E) representation of adjacency list
* @return matrix graph representation
*/
int* convertVectorGraphToMatrixGraph(VecGraph& graph) {
	auto& v = graph.first;
	auto& e = graph.second;
	int verticesNumber = v.size() - 1;
	int* matrixGraph = (int*)malloc(verticesNumber * verticesNumber * sizeof(int));
	memset(matrixGraph, 0, verticesNumber * verticesNumber * sizeof(int));

	for (int i = 0; i < verticesNumber; ++i) {
		int nEdges = v[i + 1] - v[i];
		for (int j = 0; j < nEdges; ++j) {
			matrixGraph[i * verticesNumber + e[v[i] + j]] = 1;
			matrixGraph[i + verticesNumber * e[v[i] + j]] = 1;			// symmetric matrix
		}
	}
	return matrixGraph;
}

/**
* Print graph given in matrix representation
*
* @param mtx graph in matrix representation
* @param size number of graph vertices
*/
void printMatrixGraph(int * mtx, int size) {
	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			std::cout << mtx[i*size + j] << ", ";
		}
		std::cout << std::endl;
	}
}

/**
* Prints tree given as vectors<int> of vertices and edges. Vertices vector is vector of indexes pointing to starting
* index in edge vector for its edges.
*
* @param graph vector representation (V, E) of graph 
*/
void printVectorGraph(VecGraph& graph) {
	auto& v = graph.first;
	auto& e = graph.second;
	for (int i = 0; i < v.size() - 1; ++i) {
		std::cout << i << "  ->  ";
		for (int j = v[i]; j < v[i + 1]; ++j)
			std::cout << e[j] << ", ";
		std::cout << std::endl;
	}
}

/**
* Creates vectors of indexes on which 'true' occurs. Does it for left and right vectors (left and right side of bipartial graph).
*
* @param left vector with true value on indices which represent left side of the graph
* @param right vector with true value on indices which represent right side of the graph
* @return pair of vectors of vertices of the left and right graph side
*/
std::pair<std::vector<int>, std::vector<int>> convertBoolToVertices(std::vector<bool> & left, std::vector<bool> & right) {
	std::vector<int> leftVertices, rightVertices;
	for (int i = 0; i < left.size(); ++i) {
		if (left[i])
			leftVertices.push_back(i);
	}
	for (int i = 0; i < right.size(); ++i) {
		if (right[i])
			rightVertices.push_back(i);
	}
	return std::make_pair(leftVertices, rightVertices);
};

/**
* Makes tree bipartite.
* Each tree is bipartite graph. Therefore in fact this function only divides vertices into 2 groups.
*
* @param graph is (V, E) graph representation
* @return pair of vector<bool>. First vector have 'true' on indices which are on the left side of the bipartite graph.
* Second vector has 'true' on indexes of right vertices.
*/
std::pair<std::vector<bool>, std::vector<bool>> bipartiteTree(VecGraph& graph) {
	auto& v = graph.first;
	auto& e = graph.second;
	std::vector<bool> left(e.size() + 1), right(e.size() + 1);

	if (!v.empty()) {
		left[0] = true;
		for (unsigned i = 0; i < v.size() - 1; ++i) {
			if (left[i]) {
				for (int j = v[i]; j < v[i + 1]; ++j)
					right[e[j]] = true;
			}
			else {
				for (int j = v[i]; j < v[i + 1]; ++j)
					left[e[j]] = true;
			}
		}
	}
	return std::make_pair(left, right);
}

/**
* Add random edges to matrix graph.
*
* @param graph matrix graph
* @param verticesNumber number of vertices in the graph
* @param currentEdgesNumber current number of edges in the graph
* @param desiredEdgesNumber desired number of edges in the output graph
* @return final number of edges in the output graph
*/
int addRandomEdges(int* graph, unsigned verticesNumber, int currentEdgesNumber, int desiredEdgesNumber) {
	if (currentEdgesNumber > desiredEdgesNumber || desiredEdgesNumber >  maxNumberOfEdgesInGraph(verticesNumber))
		return currentEdgesNumber;

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<> distribution(0.0, 1.0);

	int finalNumberOfEdges = currentEdgesNumber;
	auto nOfVerticesToGenerate = desiredEdgesNumber - currentEdgesNumber;
	auto potentialEdgesNumber = maxNumberOfEdgesInGraph(verticesNumber) - currentEdgesNumber;
	double probabilityOfEdgeGeneration = nOfVerticesToGenerate / (double)potentialEdgesNumber;
	for (int i = 0; i < verticesNumber; ++i) {
		for (int j = i + 1; j < verticesNumber; ++j) {
			// if edge does not exists and random number is on the good side...
			if (graph[i * verticesNumber + j] == 0 && distribution(generator) < probabilityOfEdgeGeneration) {
				graph[i * verticesNumber + j] = 1;
				graph[i + verticesNumber * j] = 1;
				++finalNumberOfEdges;
			}
		}
	}
	return finalNumberOfEdges;
}

/**
* Check edge exists between two vertices.
*
* @param graph multimap graph representation
* @param i i-th vertice of the graph
* @param j j-th vertice of the graph
* @return true if edges exists between i and j vertices, false otherwise
*/
bool isEdgeBetweenVertices(std::multimap<int, int>& graph, int i, int j) {
	auto edges_of_i = graph.equal_range(i);
	auto edges_of_j = graph.equal_range(j);

	return std::find_if(edges_of_i.first, edges_of_i.second, [&j](auto& pair) { return pair.second == j; }) != edges_of_i.second ||
		std::find_if(edges_of_j.first, edges_of_j.second, [&i](auto& pair) { return pair.second == i; }) != edges_of_j.second;
}

/**
* Get maximum number of edges in the graph.
*
* @param verticesNumber number of vertices in the graph
* @return max possible number of edges for the given number of vertices
*/
int maxNumberOfEdgesInGraph(int verticesNumber) {
	return (verticesNumber * (verticesNumber - 1)) / 2;
}

/**
* Check one graph equals the other graph
*
* @param graph1 first graph to compare given in matrix representation
* @param graph2 second graph to compare given in matrix representation
* @param verticesCount number of vertices in the graph
* @return true if graph1 is the same as graph2, false otherwise
*/
bool matrixGraphsAreEqual(int* graph1, int* graph2, int verticesCount) {
	for (int i = 0; i < verticesCount; ++i) {
		for (int j = 0; j < verticesCount; ++j) {
			if (graph1[i*verticesCount + j] != graph2[i*verticesCount + j])
				return false;
		}
	}
	return true;
}



/*********************************************************************  HELPER FUNCTIONS DEF  ********************************************************************/
/**
* Project value to given range if is out of this range.
*
* @param lowerBound minimal value
* @param upperBound maximal value
* @param value value to be projected to given range
*/
void projectValueToRange(double lowerBound, double upperBound, double & value) {
	if (value < lowerBound)
		value = lowerBound;
	else if (value > upperBound)
		value = upperBound;
}

/**
* When generating tree, this function is used to make sure graph remains connected graph each iteration.
* If desired number of edges is 0 and current vertice is the last in tree then nEdges is corrected to equal 1
* to not break the tree.
*
* @param currentVertex number of current graph vertice
* @param edgesCount number of graph edges
* @param nEdges number of edges current vertice would derive. Problem is when nEdges = 0 and potentially graph can become disconnected.
*/
void checkForConnectivity(int currentVertice, int edgesCount, double & nEdges) {
	if (static_cast<int>(nEdges) == 0 && currentVertice >= edgesCount)
		nEdges = 1;		// set minimal number of edges to make graph connected
}

std::vector<int> getRightEndVerticesOfLeftVert(std::multimap<int, int>& graph, int leftVertice, std::vector<int>& rightVertices) {
	std::vector<int> rightEndpoints;
	for (auto&& dstVertice : rightVertices) {
		if (isEdgeBetweenVertices(graph, leftVertice, dstVertice))
			rightEndpoints.push_back(dstVertice);
	}
	return rightEndpoints;
}

bool verticesCannotBeConnected(std::multimap<int, int>& graph, int i, int j, std::vector<int>& rightVertices) {
	auto edges = getRightEndVerticesOfLeftVert(graph, i, rightVertices);
	return std::find(begin(edges), end(edges), j) != end(edges);
}

bool tooFewAvailableVerticesOnTheRight(std::multimap<int, int>& graph, int leftVertice, std::vector<int>& rightVertices, int numberOfDesiredEdges) {
	int leftRightEdges = 0;
	for (auto&& dstVertice : rightVertices) {
		if (isEdgeBetweenVertices(graph, leftVertice, dstVertice))
			++leftRightEdges;
	}
	return rightVertices.size() - leftRightEdges < numberOfDesiredEdges;
}

unsigned countEdgesFromLeftVertToRightVertices(std::multimap<int, int>& graph, int leftVertice, std::vector<int>& rightVertices) {
	unsigned leftRightEdges = 0;
	for (auto&& dstVertice : rightVertices) {
		if (isEdgeBetweenVertices(graph, leftVertice, dstVertice))
			++leftRightEdges;
	}
	return leftRightEdges;
}

/**
* Check provided vec is valid shuffle pattern for isomorphic
*/
bool schuffleVectorValid(std::vector<int>& vec, int verticesNumber) {
	if (vec.size() != verticesNumber)
		return false;
	std::vector<int> copy = vec;
	std::sort(copy.begin(), copy.end());
	auto it_adj = std::adjacent_find(copy.begin(), copy.end());
	auto it_max = std::max_element(copy.begin(), copy.end());
	return copy.size() == vec.size() && it_adj == copy.end() && (*it_max) == (verticesNumber - 1);
}

/**
* Adds additional quasi-random edges betwwen left and right parts of the bipartite graph if current number of edges is less than
* desiredNofEdges.
*
* Edges are added going down the left side of the graph. For each vertice on the left we generate random number of additional edges
* and then we randomly choose which vertices on the right these edges will be connected to.
*
* @param graph bipartite graph given using vector (V, E) representation
* @param left vector which indicate left side vertices
* @param right vector which indicate right side vertices
* @param desiredNofEdges number of edges graph should have in output
* @return bipartite graph in multimap representation (vertices are keys, edges are values)
*/
std::multimap<int, int> addRandomEdgesToBipartite(VecGraph& graph,
	std::vector<bool> & left, std::vector<bool> & right, unsigned desiredNofEdges) {

	auto& v = graph.first;
	auto& e = graph.second;
	// It is easy to add new edges to multimap, where key is vertice and values are edges 
	// (vertices which derive edge together with key vertice)
	auto multimapGraph = convertVectorGraphToMultimapGraph(graph);
	if (desiredNofEdges <= e.size())	// desired number of edges has been reached already.
		return multimapGraph;

	auto lrVertices = convertBoolToVertices(left, right);
	auto leftVertices = lrVertices.first;
	auto rightVertices = lrVertices.second;

	// compute mean number of additional edges for each vertice so that finally desiredNofEdges edges exists
	double expectedMeanEdgeNumber = (desiredNofEdges - e.size()) / (double)(leftVertices.size());
	// example standard deviation, can play with it
	double standardDeviation = expectedMeanEdgeNumber * ((desiredNofEdges - e.size()) / (double)desiredNofEdges);

	std::random_device rd;
	std::mt19937 generator(rd());
	std::normal_distribution<double> normalDistribution(expectedMeanEdgeNumber, standardDeviation);
	std::uniform_int_distribution<> uniformDistribution(0, rightVertices.size() - 1);

	for (int i = 0; i < v.size() - 1; ++i) {
		if (left[i])				// add edges from left vertice to random right vertices
		{
			double nAdditionalEdges = normalDistribution(generator);
			projectValueToRange(0, rightVertices.size() - countEdgesFromLeftVertToRightVertices(multimapGraph, i, rightVertices), nAdditionalEdges);
			for (int j = 0; j < static_cast<int>(nAdditionalEdges); ++j) {
				int newEdge;
				do {
					// get which vertice on the right will be connected by new edge
					newEdge = uniformDistribution(generator);
				} while (verticesCannotBeConnected(multimapGraph, i, rightVertices[newEdge], rightVertices));
				// add new edges to i vertice
				multimapGraph.insert({ i, rightVertices[newEdge] });
			}
		}
	}
	return multimapGraph;
}

void generateShufflePattern(std::vector<int>& shufflePattern, int verticesCount) {
	shufflePattern.clear();
	shufflePattern.resize(verticesCount);
	// Fill with 0, 1, ..., verticesNumber-1
	std::iota(std::begin(shufflePattern), std::end(shufflePattern), 0);
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(shufflePattern.begin(), shufflePattern.end(), g);
}
/*****************************************************************  END OF HELPER FUNCTIONS DEF  ****************************************************************/


}
