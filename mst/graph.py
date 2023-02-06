import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str], delim = ','):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat, delim)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str, delim: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=delim)

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        self.mst = None

        # Get the number of nodes in the adjacency matrix
        num_nodes = int(np.sqrt(self.adj_mat.size))
        # Matrix check
        # Check to make sure more than one node is present
        if num_nodes == 1:
            raise ValueError("Only a single node present")
        # Check to make sure it's a square matrix:
        if self.adj_mat.shape[0] != self.adj_mat.shape[1]:
            raise ValueError("Check mat to make sure it's square")
 
        # Initialize array to store MST results, filled with zeros
        mst = np.zeros((num_nodes, num_nodes))
        # Keep track of visited nodes to make sure they are included in MST
        nodes_in_MST = [False for i in range(0, num_nodes)]
        # Create a heap that will store the edge weights
        edge_heap = []
        # Define a starting node, zero for now. Probably sub-optimal IRL and set first node to true
        u = 0
        nodes_in_MST[u] = True

        # For the first node, get the weights 
        for v in range(0, num_nodes):
            # If the edge weight is 0, do nothing with the node, it's not connected
            current_edge_weight = self.adj_mat[u][v]
            if current_edge_weight == 0:
                continue
            else:
                # Store the edge weight first, origin node, and the destination node
                heapq.heappush(edge_heap, (current_edge_weight, u, v))

        # Now that we've got things started, if there are unvisited nodes, contine to run
        while not all(nodes_in_MST): 
            # If the heap is empty, then we're out of options
            # This switch is triggered only if the prior condition that not all nodes have been visited is true
            if len(edge_heap) == 0:
                raise ValueError("Nodes do not all connect, MST does not exist")
            
            # Get the next smallest edge, current node, and the next node
            next_edge = heapq.heappop(edge_heap)
            current_node = next_edge[1]
            next_node = next_edge[2] 

            # Check to see if the node has already been visited
            if nodes_in_MST[next_node] == False:

                # Add the node to the MST
                nodes_in_MST[next_node] = True
                # Add the connection weight to the MST matrix 
                mst[current_node][next_node] = next_edge[0]
                mst[next_node][current_node] = next_edge[0]
                
                # Run it all again!
                for w in range(0, num_nodes):
                    # if the edge weight is 0, do nothing with the node, it's not connected
                    current_edge_weight = self.adj_mat[next_node][w]
                    if current_edge_weight == 0:
                        continue
                    else:
                        heapq.heappush(edge_heap, (current_edge_weight, next_node, w))
        
        # Return the completed mst
        self.mst = mst
