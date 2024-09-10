# Autonomous Haul Truck Route Optimization Project

### Overview
This project was developed as part of a Hackathon organized by the UWA Young Engineers Club, in collaboration with Rio Tinto and BCG X. The goal of the project was to **optimize the efficiency and safety of autonomous haul truck routes by improving road curve designs at intersections**. 
We used machine learning techniques to identify inefficiencies in truck movement and developed solutions to optimize the overall truck cycle time.

### Project Objective
The aim of this project was to:
1. Analyze truck movement data and identify inefficiencies caused by poorly designed road curves at intersections.
2. Classify road curves as "good" or "bad" based on curvature and angle changes.
3. Build an optimized road curve generation model to minimize sharp turns and ensure smoother truck transitions at intersections.
4. Present a data-driven solution to reduce operational costs and improve safety.

### Technology Stack
- Programming Language: **Python**
- Libraries: pandas, numpy, matplotlib, RandomForest from scikit-learn
- Data: Over 50 available datasets of truck movements provided by Rio Tinto

### Key Features
- Data Analysis: Analyzed truck movement data to calculate curvature and identify inefficiencies in road design.
- Machine Learning Model: Implemented a RandomForest machine learning model to classify road curves and optimize new designs.
- Data Visualization: Used matplotlib to visualize the truck movement patterns and identify areas for improvement.
- Optimization: Focused on reducing the variability in angle changes (Δθ) at intersections, leading to smoother truck routes and improved safety.

### How to Run the Project
1. Clone the repository:
   git clone https://github.com/yourusername/RioTinto-HaulTruck-RouteOptimization.git

2. Install required libraries:
   pip install -r requirements.txt

3. Run the Python script:
   python app.py

4. View the analysis: The results, including the optimized road curves and visualizations, will be outputted in the console or as plots.

### Project Status
Completed:
- The dataset was thoroughly analyzed to identify inefficiencies in truck movement.
- The RandomForest model was trained to classify road curves based on curvature.
- Initial optimization of road curves was conducted to improve the smoothness of truck transitions.

In Progress:
- Further refinement of the model to handle additional variables such as truck load, weather conditions, and real-time data integration.
- Expansion of the optimization model to cover different intersection types and one-way road constraints.

### Limitations
- Simplified Model: The current model only accounts for curvature and angle changes without considering other real-world factors like truck weight or external conditions (e.g., weather).
- Data Inconsistencies: Some datasets had inconsistencies, such as missing values and incomplete data, which required manual handling.
- One-Way Intersections: The model does not yet address one-way intersection scenarios, which are important for operational safety in real-world applications.
- Real-Time Application: The project currently lacks real-time data integration, meaning that any adjustments would need to be recalculated and manually implemented.

### Future Improvements
- Optimization Algorithm: Enhance the optimization algorithm to handle more complex intersection layouts, including one-way and multi-lane intersections.
- Model Accuracy: Continue training the model with additional datasets to improve classification accuracy and robustness.

### Contributors
- Jaylen Dang
- Team Members: Nam Tran, Yuxuan Zhang, Allwin Vincent

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
