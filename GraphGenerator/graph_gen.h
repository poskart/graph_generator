#pragma once

#include <vector>
#include <string>
#include <map>

namespace graph_gen {

	// Graph representation (V, E) using adjacency list form. Each element vi in V array 
	// points to the starting position of vertex i adjacency list in the array of edges E.
	// Based on graph representation from section 3.1 of article 
	// "Accelerating large graph algorithms on the GPU using CUDA" by Pawan Harish and P. J. Narayanan
	using VecGraph = std::pair<std::vector<int>, std::vector<int>>;


	/***********************************************************************  GRAPH GENERATION  **********************************************************************/

	VecGraph generateVectorTree(int verticesCount, double expectedMeanEdgeNumber, double standardDeviation);

	int* generateMatrixTree(int verticesCount, double expectedMeanEdgeNumber, double standardDeviation);


	VecGraph generateVectorGraph(int verticesCount, int approxEdgesCount, double exptdMeanEdgeNumberForTreeVertice, double stdDeviation);

	int* generateMatrixGraph(int verticesCount, int approxEdgesCount, double exptdMeanEdgeNumberForTreeVertice, double stdDeviation, int* finalEdgesCount);


	VecGraph generateBipartiteVectorGraph(int verticesCount, int approxEdgesCount, double exptdMeanEdgeNumberForTreeVertice, double stdDeviation);

	int* generateBipartiteMatrixGraph(int verticesCount, int approxEdgesCount, double exptdMeanEdgeNumberForTreeVertice, double stdDeviation);


	VecGraph generateIsomorphicVectorGraph(VecGraph& graph, std::vector<int> shufflePattern = {});

	int* generateIsomorphicMatrixGraph(int* graph, int verticesCount, int edgesCount = -1, std::vector<int> shufflePattern = {});


	
	/***********************************************************************  GRAPH CONVERSIONS  **********************************************************************/

	VecGraph convertMultimapGraphToVectorGraph(std::multimap<int, int> & graph, int verticesCount);

	VecGraph convertMatrixGraphToVectorGraph(int* graph, int verticesNumber, int edgesNumber = -1);

	std::multimap<int, int> convertVectorGraphToMultimapGraph(VecGraph&);

	int* convertVectorGraphToMatrixGraph(VecGraph&);



	/************************************************************************  GRAPH DISPLAY  ************************************************************************/

	void printMatrixGraph(int * mtx, int size);

	void printVectorGraph(VecGraph& graph);



	/***********************************************************************  HELPER FUNCTIONS  **********************************************************************/

	std::pair<std::vector<int>, std::vector<int>> convertBoolToVertices(std::vector<bool> & left, std::vector<bool> & right);

	std::pair<std::vector<bool>, std::vector<bool>> bipartiteTree(VecGraph&);

	int addRandomEdges(int* graph, unsigned verticesNumber, int currentEdgesNumber, int desiredEdgesNumber);

	bool isEdgeBetweenVertices(std::multimap<int, int>& graph, int i, int j);

	int maxNumberOfEdgesInGraph(int verticesNumber);

	bool matrixGraphsAreEqual(int* graph1, int* graph2, int verticesNumber);
}
