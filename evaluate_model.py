# evaluate_model.py

import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

# Function to test the model and measure performance
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_time = 0  # To measure total processing time for all graphs

    with torch.no_grad():
        for data in loader:
            # Measure start time
            start_time = time.time()
            
            # Perform forward pass
            out = model(data)
            
            # Measure end time and accumulate processing time
            end_time = time.time()
            total_time += (end_time - start_time)
            
            # Get predictions and true labels
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(data.y.view(-1).cpu().numpy())  # Reshape target for consistency
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100  # Convert to percentage
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100  # Convert to percentage
    avg_processing_time = (total_time / len(loader.dataset)) * 1000  # Convert to ms per graph

    return accuracy, f1, avg_processing_time

# Main function to evaluate the proposed model
if __name__ == "__main__":
    # Load the model
    model = torch.load('H/saved_model/saved_model.pth')  # Ensure this is the correct path and file extension
    model.eval()  # Set the model to evaluation mode

    # Load the test loader
    from F:\IPCV_GUI\H\og\Andromeda001_png.graphml import test_loader  # Replace with your actual data loader file

    # Evaluate the proposed model
    test_accuracy, test_f1, avg_time = evaluate_model(model, test_loader)

    # Print results
    print(f"Model Performance on Test Set:")
    print(f"Accuracy: {test_accuracy:.2f}%")
    print(f"F1 Score: {test_f1:.2f}%")
    print(f"Average Processing Time per Graph: {avg_time:.2f} ms")

    # Save results to a dictionary for comparison
    proposed_model_results = {
        "Accuracy (%)": test_accuracy,
        "F1 Score (%)": test_f1,
        "Processing Time (ms/graph)": avg_time
    }

    # Print the results dictionary
    print("Proposed Model Results:", proposed_model_results)