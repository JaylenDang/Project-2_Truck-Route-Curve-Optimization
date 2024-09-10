import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from flask import Flask, request, jsonify,render_template

# Provide the full path to the file using raw string
file_path = "data/B4_truck_movements_10.csv"

# Load the dataset using comma as separator
df = pd.read_csv(file_path, sep=',', header=None)
df.columns = ['TruckID', 'Timestamp', 'X', 'Y']

# Placeholder for intersection edges and adjacent roads (update with actual data)
intersection_edges = np.array([
    [(0, 0), (0, 100)], 
    [(0, 100), (100, 100)], 
    [(100, 100), (100, 0)], 
    [(100, 0), (0, 0)]
])
adjacent_roads = [[(0, 0), (0, 50)], [(0, 100), (50, 100)]]

# Function to calculate distance to edge
def distance_to_edge(x, y, edges):
    min_distance = float('inf')
    for edge in edges:
        edge_start, edge_end = np.array(edge[0]), np.array(edge[1])
        distance = np.abs(np.cross(edge_end - edge_start, edge_start - np.array([x, y]))) / np.linalg.norm(edge_end - edge_start)
        if distance < min_distance:
            min_distance = distance
    return min_distance

# Function to check smooth entry and exit
def is_smooth_entry_exit(start_point, end_point, roads):
    for road in roads:
        road_start, road_end = np.array(road[0]), np.array(road[1])
        if np.allclose(start_point, road_start, atol=1) and np.allclose(end_point, road_end, atol=1):
            return True
    return False

# Function to calculate distance between curves
def distance_between_curves(curve1, curve2):
    min_distance = float('inf')
    for point1 in curve1:
        for point2 in curve2:
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            if distance < min_distance:
                min_distance = distance
    return min_distance

# Function to calculate curvature
def calculate_curvature(df):
    df['XDiff'] = df['X'].diff()
    df['YDiff'] = df['Y'].diff()
    df['Direction'] = np.arctan2(df['YDiff'], df['XDiff'])
    df['Curvature'] = df['Direction'].diff().abs()
    df['Curvature'] = df['Curvature'].fillna(0)
    return df

# Generate labels based on criteria
labels = []
for truck_id in df['TruckID'].unique():
    truck_df = df[df['TruckID'] == truck_id].copy()
    truck_df = calculate_curvature(truck_df)
    
    start_point = truck_df.iloc[0][['X', 'Y']].values
    end_point = truck_df.iloc[-1][['X', 'Y']].values
    within_bounds = all(truck_df.apply(lambda row: distance_to_edge(row['X'], row['Y'], intersection_edges) >= 6.92, axis=1))
    smooth_entry_exit = is_smooth_entry_exit(start_point, end_point, adjacent_roads)
    maintain_distance = distance_between_curves(truck_df[['X', 'Y']].values, truck_df[['X', 'Y']].values) >= 15.94
    curvature_ok = all(truck_df['Curvature'] < np.pi / 4)  # Example threshold for curvature
    
    is_good_curve = within_bounds, smooth_entry_exit, maintain_distance, curvature_ok
    labels.extend([1 if all(is_good_curve) else 0] * len(truck_df))

# Add labels to the DataFrame
df['Label'] = labels

# Convert Timestamp to numerical values for modeling
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d %b %Y %H:%M:%S:%f')

# Split the data into features and labels
features = df[['X', 'Y', 'Timestamp']]
labels = df['Label']

# Convert Timestamp to numerical values
features['Timestamp'] = features['Timestamp'].astype(int) / 10**9

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Flask Application to Serve the Model
app = Flask(__name__)

# Function to classify a curve
def classify_curve(data):
    df = pd.DataFrame(data)
    df = calculate_curvature(df)
    
    start_point = df.iloc[0][['X', 'Y']].values
    end_point = df.iloc[-1][['X', 'Y']].values
    within_bounds = all(df.apply(lambda row: distance_to_edge(row['X'], row['Y'], intersection_edges) >= 6.92, axis=1))
    smooth_entry_exit = is_smooth_entry_exit(start_point, end_point, adjacent_roads)
    maintain_distance = distance_between_curves(df[['X', 'Y']].values, df[['X', 'Y']].values) >= 15.94
    curvature_ok = all(df['Curvature'] < np.pi / 4)  # Example threshold for curvature
    
    is_good_curve = within_bounds, smooth_entry_exit, maintain_distance, curvature_ok
    label = 1 if all(is_good_curve) else 0
    return label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    label = classify_curve(data)
    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)