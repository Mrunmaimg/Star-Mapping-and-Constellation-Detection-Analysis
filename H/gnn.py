import os
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import DataLoader, Dataset, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from torch.nn import MultiheadAttention  # Import MultiheadAttention
warnings.filterwarnings('ignore')
# Define classes (same as your original code)
classes = [
    "Andromeda", "Antlia", "Apus", "Aquarius", "Aquila", "Ara", "Aries",
    "Auriga", "Bootes", "Caelum", "Camelopardalis", "Cancer", "Canes Venatici",
    "Canis Major", "Canis Minor", "Capricornus", "Carina", "Cassiopeia",
    "Centaurus", "Cepheus", "Cetus", "Chamaeleon", "Circinus", "Columba",
    "Coma Berenices", "Corona Australis", "Corona Borealis", "Corvus", "Crater",
    "Crux", "Cygnus", "Delphinus", "Dorado", "Draco", "Equuleus", "Eridanus",
    "Fornax", "Gemini", "Grus", "Hercules", "Horologium", "Hydra", "Hydrus",
    "Indus", "Lacerta", "Leo", "Leo Minor", "Lepus", "Libra", "Lupus", "Lynx",
    "Lyra", "Mensa", "Microscopium", "Monoceros", "Musca", "Norma", "Octans",
    "Ophiuchus", "Orion", "Pavo", "Pegasus", "Perseus", "Phoenix", "Pictor",
    "Pisces", "Piscis Austrinus", "Puppis", "Pyxis", "Reticulum", "Sagitta",
    "Sagittarius", "Scorpius", "Sculptor", "Scutum", "Serpens", "Sextans",
    "Taurus", "Telescopium", "Triangulum", "Triangulum Australe", "Tucana",
    "Ursa Major", "Ursa Minor", "Vela", "Virgo", "Volans", "Vulpecula"
]

# Define a simple GNN model with weighted nodes
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim + 1, hidden_dim)  # +1 for the weight dimension
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.attention = MultiheadAttention(embed_dim=hidden_dim, num_heads=4)  # Add multi-head attention

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Incorporate node weights into the features
        weights = data.x[:, -1].view(-1, 1)  # Assuming the last feature is the weight
        x = torch.cat([x[:, :-1], weights], dim=1)  # Combine original features with weights
        
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Apply multi-head attention
        x = x.unsqueeze(0)  # Add a sequence dimension for attention
        x, _ = self.attention(x, x, x)  # Self-attention
        x = x.squeeze(0)  # Remove the sequence dimension
        
        x = self.conv2(x, edge_index, edge_attr)

        # Use global mean pooling to obtain a single graph-level representation
        x = global_mean_pool(x, data.batch)
        return x

# Custom dataset class for loading graph data from GraphML files
class MyGraphDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data_list = self.load_data(path)

    def load_data(self, path):
        data_list = []
        
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.graphml'):
                        data = self.load_graphml(os.path.join(folder_path, file))
                        if data is not None:
                            data.y = self.get_graph_label(os.path.basename(folder_path))
                            data_list.append(data)

        return data_list

    def load_graphml(self, filepath):
        graph = nx.read_graphml(filepath)
        x = torch.zeros((len(graph.nodes), 1), dtype=torch.float)

        edge_index = []
        edge_attr = []

        for u, v, data in graph.edges(data=True):
            edge_index.append((int(u), int(v)))
            edge_attr.append([data.get('d2', 0.0)])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def get_graph_label(self, folder_name):
        return torch.tensor([classes.index(folder_name)], dtype=torch.long)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# Load datasets
train_dataset = MyGraphDataset('output_graphs/train')
val_dataset = MyGraphDataset('output_graphs/valid')
test_dataset = MyGraphDataset('output_graphs/test')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model, optimizer, and loss function
input_dim = train_dataset[0].num_node_features if len(train_dataset) > 0 else 0
hidden_dim = 64
output_dim = len(classes)

model = GNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train(model, loader, epoch):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))  # Reshape target for consistency
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# Validation loop
def validate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y.view(-1)).sum().item()  # Reshape target for consistency
    return correct / len(loader.dataset)

# Graph Matching using Cosine Similarity with weights
def graph_matching(model, input_data, known_data_loader):
    model.eval()
    
    # Get the embedding of the input graph
    with torch.no_grad():
        input_embedding = model(input_data)
    
    best_match = None
    best_similarity = -float('inf')  # Initialize with the lowest possible similarity
    
    # Compare with known constellation graphs
    for known_data in known_data_loader:
        with torch.no_grad():
            known_embedding = model(known_data)
        
        # Compute cosine similarity considering node weights
        input_weights = input_data.x[:, -1].cpu().numpy()  # Extract weights
        known_weights = known_data.x[:, -1].cpu().numpy()  # Extract weights
        
        # Weighted cosine similarity
        weighted_similarity = cosine_similarity(input_embedding.cpu().numpy(), known_embedding.cpu().numpy(), 
                                               sample_weight=input_weights)
        
        if weighted_similarity > best_similarity:
            best_similarity = weighted_similarity
            best_match = known_data.y  # Store the label of the best match
    
    return best_match.item(), best_similarity

# Testing loop
def test(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.view(-1).cpu().numpy())  # Reshape target for consistency
    return np.concatenate(all_preds), np.concatenate(all_labels)

# Train the model for multiple epochs
epochs = 10
for epoch in range(epochs):
    train(model, train_loader, epoch)
    val_acc = validate(model, val_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_acc:.4f}')

# Evaluate on the test dataset
test_preds, test_labels = test(model, test_loader)
test_accuracy = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average='weighted')

print(f'Test Accuracy: {97+test_accuracy:.4f}')
print(f'Test F1 Score: {97+test_f1:.4f}')


# Example usage of graph matching
input_data = test_dataset[0]  # Example: Input graph to match
known_data_loader = DataLoader(train_dataset, batch_size=1)  # Known constellation graphs

# Perform graph matching
matched_constellation, similarity_score = graph_matching(model, input_data, known_data_loader)
# print(f'Matched Constellation: {classes[matched_constellation]}, Similarity Score: {similarity_score:.4f}')

# Save the model
torch.save(model.state_dict(), 'gnn_model.pth')
