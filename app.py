from flask import Flask, request, jsonify, render_template
import os
import time
import cv2
import numpy as np
import networkx as nx

app = Flask(__name__)

# Define upload folder for the images
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def process_maze(image_path):
    """Solve the maze and return solved image path and time taken."""
    start_time = time.time()
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    rows, cols = binary.shape
    graph = nx.Graph()
    
    for y in range(rows):
        for x in range(cols):
            if binary[y, x] == 255:
                if x > 0 and binary[y, x - 1] == 255:
                    graph.add_edge((y, x), (y, x - 1))
                if y > 0 and binary[y - 1, x] == 255:
                    graph.add_edge((y, x), (y - 1, x))

    start, end = (0, 0), (rows - 1, cols - 1)
    if start not in graph or end not in graph:
        return None, "No valid path found"

    try:
        path = nx.shortest_path(graph, source=start, target=end)
    except nx.NetworkXNoPath:
        return None, "No valid path"

    solved_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for y, x in path:
        solved_img[y, x] = (0, 0, 255)  # Red path

    solved_path = os.path.join(UPLOAD_FOLDER, "solved_maze.png")
    cv2.imwrite(solved_path, solved_img)

    time_taken = round(time.time() - start_time, 2)
    return solved_path, time_taken

@app.route('/solve', methods=['POST'])
def solve():
    if 'maze_image' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})

    file = request.files['maze_image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})

    file_path = os.path.join(UPLOAD_FOLDER, "maze.png")
    file.save(file_path)

    solved_path, time_taken = process_maze(file_path)
    if solved_path is None:
        return jsonify({"success": False, "error": "Maze could not be solved"})

    return jsonify({"success": True, "image_url": "/static/solved_maze.png", "time_taken": time_taken})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
