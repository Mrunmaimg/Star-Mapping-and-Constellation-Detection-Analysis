import os
import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class ConstellationDetection:
    def __init__(self, output_folder):
        self.output_folder = output_folder

    # Extract keypoints from an image using ORB
    @staticmethod
    def extract_keypoints(image_path):
        image = cv2.imread(image_path, 0)  # Read the image in grayscale
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        orb = cv2.ORB_create(nfeatures=20)  # Set the maximum number of features
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
        if keypoints is None or len(keypoints) == 0:
            print(f"No keypoints detected in image: {image_path}")
            return None
        
        return keypoints

       # Visualize and save keypoints on the image
    @staticmethod
    def visualize_keypoints(image_path, keypoints):
        image = cv2.imread(image_path)  # Read the original image in color
        image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Convert the image from BGR to RGB format
        image_with_keypoints = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
        
        # Display the image with keypoints
        plt.figure(figsize=(10, 8))  # Optional: adjust the size of the figure
        plt.imshow(image_with_keypoints)
        
        plt.title("Keypoints")
        plt.axis('off')  # Hide axes

        # Save the figure to a file instead of closing it immediately
        plt.savefig(os.path.join(os.path.dirname(image_path), "keypoints_" + os.path.basename(image_path)), bbox_inches='tight')
        plt.show(block=False) 
        plt.pause(0.1)
        plt.close()# Show the image

    # Build a graph from keypoints, connecting them with edges based on distances
    @staticmethod
    def build_graph_from_keypoints(keypoints):
        graph = nx.Graph()
        
        # Add keypoints as nodes in the graph, storing x, y coordinates as separate attributes
        for i, kp in enumerate(keypoints):
            pos = (float(kp.pt[0]), float(kp.pt[1]))
            graph.add_node(i, pos_x=pos[0], pos_y=pos[1])  # Store x and y as separate float attributes
        
        # Connect nodes with edges based on distances, storing distance as float
        for i in range(len(keypoints)):
            for j in range(i + 1, len(keypoints)):
                distance = np.linalg.norm(np.array(keypoints[i].pt) - np.array(keypoints[j].pt))
                graph.add_edge(i, j, weight=float(distance))  # Ensure distance is a float
        
        return graph

    # Save the graph to a .graphml file
    @staticmethod
    def save_graph(graph, output_path):
        try:
            # Attempt to save as GraphML
            nx.write_graphml(graph, output_path)
            print(f"Graph saved to {output_path}")
        except Exception as e:
            print(f"Failed to save graph {output_path}: {str(e)}")

    # Process a single image and convert it to a graph
    def process_image(self, image_path):
        keypoints = self.extract_keypoints(image_path)
        
        if keypoints is not None:
            graph = self.build_graph_from_keypoints(keypoints)
            graph_output_path = os.path.join(self.output_folder, f"{os.path.basename(image_path).split('.')[0]}.graphml")
            self.save_graph(graph, graph_output_path)
            print(f"Processed {os.path.basename(image_path)}, {len(keypoints)} keypoints detected.")
            # self.visualize_keypoints(image_path, keypoints)  # Optional: visualize the keypoints
if __name__ == '__main__':
    output_folder = 'og'  # Folder where graphs will be saved

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create an instance of ConstellationDetection
    constellation_detector = ConstellationDetection(output_folder)

    # Specify the path to the image you want to process
    image_path = 'data/train/Andromeda001_png.rf.9cf280ab89ad711a8f9672c049663667.jpg'  # Replace with the actual path to your image

    # Process the single image
    constellation_detector.process_image(image_path)
