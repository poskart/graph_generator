#include <iostream>
#include <ios>
#include "graph_gen.h"

int main() {
	// Generate and print quasi-random bipartite graph
	auto b_graph = graph_gen::generateBipartiteVectorGraph(20, 60, 3, 0.2);
	std::cout << "Bipartite graph:" << std::endl;
	graph_gen::printVectorGraph(b_graph);
	std::cout << "|V| = " << b_graph.first.size() - 1 << 
		"\t\t|E| = " << b_graph.second.size() << std::endl << std::endl;

	// Generate and print quasi random graph g1 and second graph g2 isomorphic to g1
	auto g1 = graph_gen::generateVectorGraph(5, 10, 2.2, 0.5);
	auto g2 = graph_gen::generateIsomorphicVectorGraph(g1);
	
	std::cout << "Graph g1:" << std::endl;
	graph_gen::printVectorGraph(g1);
	std::cout << "\nGraph g2:" << std::endl;
	graph_gen::printVectorGraph(g2);
}