# Star Mapping and Constellation Detection Analysis

This project implements an advanced system for detecting and analyzing constellations using **Modified Graph Neural Networks (MGNN)** and **Graph Matching Networks (GMN)**. It combines computer vision, graph theory, and neural networks for precise and efficient constellation identification.

---

## **Key Features**
- **Dynamic Graph Construction**:
  - **Keypoint Detection**: Uses ORB (Oriented FAST and Rotated BRIEF) for identifying star positions.
  - **Graph Representation**: Spatial relationships captured via Delaunay Triangulation.
  - Output stored in **GraphML format** for downstream processing.

- **Feature Encoding**:
  - Node attributes based on star intensity and spatial arrangement.
  - Edge attributes derived from Euclidean distances.

- **Neural Network Analysis**:
  - Trained **GNN** for constellation classification using graph convolution layers.
  - Incorporates **graph matching** for updating and detecting new constellations.

- **Robust and Efficient**:
  - Handles incomplete and noisy star maps with cosine similarity-based graph matching.
  - High computational efficiency with low average processing time (~0.05 sec).

---

## **Performance**
| **Metric**                    | **Value** |
|--------------------------------|-----------|
| Average Graph Construction Time | 0.05 sec |
| Classification Accuracy        | 94.3%     |
| Cosine Similarity (Noisy Data) | 87%       |
| F1-Score                       | 0.91      |

---

## **Methodology**
### 1. **Star Map Processing**
- Extracts keypoints using the **ORB algorithm**.
- Filters noise for accurate detection.

### 2. **Graph Construction**
- Uses **Delaunay Triangulation** to form a graph.
- Nodes represent stars; edges represent spatial distances.

### 3. **Learning and Classification**
- Trained GNN to classify constellation graphs.
- Graph-level pooling for compact feature representation.

### 4. **Dynamic Graph Matching**
- Updates with new star maps using cosine similarity of graph embeddings.

---

## **Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/Mrunmaimg/Star-Mapping-and-Constellation-Detection-Analysis.git
   ```
2. Navigate to the directory:
   ```bash
   cd Star-Mapping-and-Constellation-Detection-Analysis
   ```
3. Run the main file:
   ```bash
   python one.py
   ```

---

## **Future Enhancements**
- Add attention mechanisms for sparse graph classification.
- Extend to 3D star configurations.
- Parameter optimization for better adaptability.


---

