#include <iostream>
#include <list>
#include <vector>
#include "graph.h"

int main()
{
    OpeJobProc graph(5, 1);

    graph.addEdge(0, 1);
    graph.addEdge(0, 2);
    graph.addEdge(1, 3);
    graph.addEdge(2, 3);
    graph.addEdge(3, 4);

    graph.printGraph();

    graph.updateCumulative();

    graph.printCumulative();

    return 0;
}