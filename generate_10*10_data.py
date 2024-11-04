import numpy as np

grid_size = 10


nodes = [(i, j) for i in range(grid_size) for j in range(grid_size)]
edges = []


for i in range(grid_size):
    for j in range(grid_size):
        node = (i, j)

        if j < grid_size - 1:
            right_node = (i, j + 1)
            length = np.random.randint(1, 20) 
            edges.append((node, right_node, length))
        

        if i < grid_size - 1:
            bottom_node = (i + 1, j)
            length = np.random.randint(1, 20) 
            edges.append((node, bottom_node, length))
nodeID = [[i,nodes[i]] for i in range(len(nodes))]
edgeID = [[i,edges[i]] for i in range(len(edges))]
edge_node = []
for edge in edgeID:
    data = [edge]
    edge_node.append(data)
print("Nodes:")
print(nodes)

print("\nEdges (node1, node2, length):")
print(edges)
