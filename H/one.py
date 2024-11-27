import cv2
import tensorflow as tf
import numpy as np
import json
import os
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from preprocess import ConstellationDetection

# Load the TensorFlow SavedModel
saved_model_dir = 'saved_model'  # Path to your SavedModel directory
loaded_model = tf.saved_model.load(saved_model_dir)

# Get the function for serving/inference
infer = loaded_model.signatures['serving_default']

# Function to preprocess the image (resize, normalize, etc.)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (640, 640))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

# Function to predict the constellation class
def predict_constellation(image_path):
    img = preprocess_image(image_path)
    input_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    predictions = infer(x=input_tensor)

    if isinstance(predictions, tuple):
        predicted_scores = predictions[0]
    else:
        predicted_scores = predictions['output_0']

    predicted_class_index = np.argmax(predicted_scores, axis=-1)
    predicted_scores_np = predicted_scores.numpy().flatten()
    
    return predicted_class_index, predicted_scores_np

# Function to load graph image
def load_graph_image(predicted_class_name, graph_folder):
    graph_image_path = os.path.join(graph_folder, f"{predicted_class_name}.png")
    if os.path.exists(graph_image_path):
        graph_img = cv2.imread(graph_image_path)
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(graph_img)
    else:
        print(f"Graph image for {predicted_class_name} not found!")
        return None

# Function to load and get information from info.json
def get_class_info(predicted_class_name, info_json_path):
    with open(info_json_path, 'r') as f:
        info_data = json.load(f)
    return info_data.get(predicted_class_name, "No information available.")

# Function to select an image file
def select_image():
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*"))
    )
    if image_path:
        print(f"Selected image: {image_path}")  # Debugging line
        main(image_path, graph_folder, info_json_path)

# Main function
def main(image_path, graph_folder, info_json_path):
    constellation_detector = ConstellationDetection("g")
    constellation_detector.process_image(image_path)
    predicted_class_index, predicted_scores = predict_constellation(image_path)

    classes = [
        "Andromeda", "Aquarius", "Aquila", "Auriga", "Canis Major",
        "Capricornus", "Cassiopeia", "Cetus", "Columba", "Cygnus",
        "Draco", "Gemini", "Grus", "Hercules", "Hydra",
        "Leo", "Lepus", "Lupus", "Orion", "Pavo",
        "Pegasus", "Phoenix", "Pisces", "Piscis Austrinus", "Puppis",
        "Scorpius", "Taurus", "Ursa Major", "Ursa Minor", "Vela"
    ]

    predicted_class_index = predicted_class_index.flatten()[0] % len(classes)
    predicted_class_name = classes[predicted_class_index]
    class_info = get_class_info(predicted_class_name, info_json_path)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_image = Image.fromarray(img)

    graph_image = load_graph_image(predicted_class_name, graph_folder)

    display_results(original_image, graph_image, predicted_class_name, class_info)

# Function to display results in a GUI with scrollbars
def display_results(original_image, graph_image, predicted_class_name, class_info):
    root = tk.Tk()
    root.title("Constellation Detection And Analysis")
    root.geometry("800x600")

    # Create a canvas for scrolling
    canvas = tk.Canvas(root)
    scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scroll_x = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    
    scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
    scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    # Create a frame to hold the images and information
    frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor='center')  # Center the frame

    # Bind the frame to the canvas to enable scrolling
    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    frame.bind("<Configure>", on_frame_configure)

    # Use grid to center the content
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_rowconfigure(1, weight=1)
    frame.grid_rowconfigure(2, weight=1)
    frame.grid_rowconfigure(3, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    # Create a frame for the original image
    img_frame = ttk.Frame(frame)
    img_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
    
    # Display original image centered
    img_display = ImageTk.PhotoImage(original_image)
    img_original = ttk.Label(img_frame, image=img_display)
    img_original.image = img_display  # Keep a reference
    img_original.pack(padx=10, pady=10)  # Center the image

    # Create a frame for the graph image
    graph_frame = ttk.Frame(frame)
    graph_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

    # Display graph image centered
    if graph_image is not None:
        graph_display = ImageTk.PhotoImage(graph_image)
        img_graph = ttk.Label(graph_frame, image=graph_display)
        img_graph.image = graph_display  # Keep a reference
        img_graph.pack(padx=10, pady=10)  # Center the graph image

    # Create a frame for text information
    info_frame = ttk.Frame(frame)
    info_frame.grid(row=2, column=0, padx=20, pady=10, sticky='nsew')  # Center the info frame

    # Display predicted class name
    class_label = ttk.Label(info_frame, text=f"Predicted Constellation: {predicted_class_name}", font=("Helvetica", 16, "bold"))
    class_label.pack(pady=(10, 0))  # Center the class label

    # Create a text widget for class info
    info_text = tk.Text(info_frame, wrap=tk.WORD, height=10, width=60, font=("Helvetica", 12))
    info_text.insert(tk.END, f"Information:\n\n")
    for key, value in class_info.items():
        info_text.insert(tk.END, f"{key.capitalize()}: {value}\n\n")
    info_text.config(state=tk.DISABLED)  # Make it read-only
    info_text.pack(pady=(10, 0))  # Center the text widget

    # Create a button frame for actions
    button_frame = ttk.Frame(info_frame)
    button_frame.pack(pady=(10, 0))

    # Exit button
    exit_button = ttk.Button(button_frame, text="Exit", command=root.quit)
    exit_button.pack(side=tk.LEFT, padx=5)  # Center the exit button

    # Add an option to select an image
    select_button = ttk.Button(button_frame, text="Select Another Image", command=select_image)
    select_button.pack(side=tk.LEFT, padx=5)  # Center the select button

    root.mainloop()

# Example usage
if __name__ == '__main__':
    graph_folder = 'Normalised_Templates'  # Folder containing graph images
    info_json_path = 'constellation_info.json'  # Path to the info.json file
    select_image()  # Start the image selection process
