import networkx as nx
import torch
from torch_geometric.data import Data

def extract_graph_data(graphml_file):
    # Load the graph from GraphML file
    graph = nx.read_graphml(graphml_file)

    # Initialize node features (assuming no features are defined)
    x = torch.zeros(len(graph.nodes), dtype=torch.float)

    # Extract edge indices and edge attributes
    edge_index = []
    edge_attr = []

    for u, v, data in graph.edges(data=True):
        edge_index.append((int(u), int(v)))  # Convert to integer indices
        
        # Check if 'd2' exists in the edge data
        if 'd2' in data:
            edge_attr.append([data['d2']])  # Extract the edge attribute
        else:
            edge_attr.append([0.0])  # Default to 0.0 if 'd2' doesn't exist

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Initialize labels (assuming no labels are defined)
    y = torch.zeros(len(graph.nodes), dtype=torch.long)

    # Create the Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data



# Example usage
graphml_file_path = 'output_graphs/test/Sagitta/Sagitta_png.graphml'
data = extract_graph_data(graphml_file_path)

# Print the extracted data
print(data)
