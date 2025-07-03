"""
Student Exercises for Fuel Efficiency Predictor
==============================================

These exercises will help you deepen your understanding of machine learning concepts.
Complete them in order, as they build upon each other.

Prerequisites: Complete the main fuel_efficiency_predictor.py first.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from fuel_efficiency_predictor import FuelEfficiencyPredictor

class StudentExercises:
    """
    Interactive exercises for learning machine learning concepts.
    """
    
    def __init__(self):
        self.predictor = FuelEfficiencyPredictor()
        self.predictor.load_data()
        self.predictor.prepare_data()
        
    def exercise_1_feature_importance(self):
        """
        EXERCISE 1: Feature Importance Analysis
        
        Goal: Understand which features are most important for prediction.
        
        Tasks:
        1. Train a Random Forest model
        2. Plot feature importance
        3. Remove the least important feature and retrain
        4. Compare performance
        """
        print("="*60)
        print("EXERCISE 1: FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        print("\n🎯 YOUR TASK:")
        print("1. Train a Random Forest model on the data")
        print("2. Get feature importances using model.feature_importances_")
        print("3. Create a bar plot of feature importance")
        print("4. Remove the least important feature and retrain")
        print("5. Compare R² scores before and after")
        
        print("\n💡 SOLUTION TEMPLATE:")
        print("# Step 1: Train Random Forest")
        print("rf_model = RandomForestRegressor(n_estimators=100, random_state=42)")
        print("rf_model.fit(self.predictor.X_train, self.predictor.y_train)")
        print("# Step 2-5: Complete the implementation!")
        
    def exercise_2_hyperparameter_tuning(self):
        """
        EXERCISE 2: Hyperparameter Tuning
        
        Goal: Learn how model parameters affect performance.
        
        Tasks:
        1. Try different n_estimators values for Random Forest
        2. Plot performance vs n_estimators
        3. Find the optimal value
        """
        print("="*60)
        print("EXERCISE 2: HYPERPARAMETER TUNING")
        print("="*60)
        
        print("\n🎯 YOUR TASK:")
        print("1. Test Random Forest with n_estimators = [10, 50, 100, 200, 500]")
        print("2. Record R² score for each value")
        print("3. Plot performance vs n_estimators")
        print("4. Identify the optimal value")
        
        # SOLUTION TEMPLATE (uncomment and complete):

        n_estimators_values = [10, 50, 100, 200, 500]
        r2_scores = []
        
        for n_est in n_estimators_values:
            # Train model with current n_estimators
            rf_model = RandomForestRegressor(n_estimators=n_est, random_state=42)
            rf_model.fit(self.predictor.X_train, self.predictor.y_train)
            
            # Evaluate
            y_pred = rf_model.predict(self.predictor.X_test)
            r2 = r2_score(self.predictor.y_test, y_pred)
            r2_scores.append(r2)
            
            print(f"n_estimators={n_est}: R² = {r2:.3f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(n_estimators_values, r2_scores, 'bo-')
        plt.xlabel('Number of Estimators')
        plt.ylabel('R² Score')
        plt.title('Random Forest Performance vs Number of Estimators')
        plt.grid(True)
        plt.show()
        
        # Find optimal
        best_idx = np.argmax(r2_scores)
        best_n_estimators = n_estimators_values[best_idx]
        best_r2 = r2_scores[best_idx]
        
        print(f"\nOptimal n_estimators: {best_n_estimators} (R² = {best_r2:.3f})")

        
    def exercise_3_feature_engineering(self):
        """
        EXERCISE 3: Feature Engineering
        
        Goal: Create new features to improve model performance.
        
        Tasks:
        1. Create a power-to-weight ratio feature
        2. Create an efficiency score feature
        3. Train models with new features
        4. Compare performance
        """
        print("="*60)
        print("EXERCISE 3: FEATURE ENGINEERING")
        print("="*60)
        
        print("\n🎯 YOUR TASK:")
        print("1. Create 'power_to_weight' = horsepower / weight")
        print("2. Create 'efficiency_score' = acceleration / displacement")
        print("3. Add these features to your dataset")
        print("4. Train models and compare R² scores")
        
        # SOLUTION TEMPLATE (uncomment and complete):

        # Create new features
        data_enhanced = self.predictor.data.copy()
        
        # Power-to-weight ratio
        data_enhanced['power_to_weight'] = data_enhanced['horsepower'] / data_enhanced['weight']
        
        # Efficiency score
        data_enhanced['efficiency_score'] = data_enhanced['acceleration'] / data_enhanced['displacement']
        
        # Prepare enhanced dataset
        feature_names_enhanced = [col for col in data_enhanced.columns if col != 'mpg']
        X_enhanced = data_enhanced[feature_names_enhanced]
        y = data_enhanced['mpg']
        
        X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(
            X_enhanced, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest with enhanced features
        rf_enhanced = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_enhanced.fit(X_train_enh, y_train_enh)
        y_pred_enhanced = rf_enhanced.predict(X_test_enh)
        r2_enhanced = r2_score(y_test_enh, y_pred_enhanced)
        
        # Compare with original
        rf_original = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_original.fit(self.predictor.X_train, self.predictor.y_train)
        y_pred_original = rf_original.predict(self.predictor.X_test)
        r2_original = r2_score(self.predictor.y_test, y_pred_original)
        
        print(f"Original features R²: {r2_original:.3f}")
        print(f"Enhanced features R²: {r2_enhanced:.3f}")
        print(f"Improvement: {r2_enhanced - r2_original:.3f}")
        
        # Feature importance with new features
        importance_enhanced = rf_enhanced.feature_importances_
        
        plt.figure(figsize=(12, 6))
        plt.barh(feature_names_enhanced, importance_enhanced)
        plt.xlabel('Importance')
        plt.title('Feature Importance (Enhanced Dataset)')
        plt.tight_layout()
        plt.show()

        
    def exercise_4_cross_validation(self):
        """
        EXERCISE 4: Cross-Validation
        
        Goal: Learn robust model evaluation techniques.
        
        Tasks:
        1. Implement 5-fold cross-validation
        2. Compare Linear Regression and Random Forest
        3. Calculate mean and standard deviation of scores
        """
        print("="*60)
        print("EXERCISE 4: CROSS-VALIDATION")
        print("="*60)
        
        print("\n🎯 YOUR TASK:")
        print("1. Use cross_val_score with cv=5")
        print("2. Compare Linear Regression and Random Forest")
        print("3. Calculate mean ± std for each model")
        print("4. Determine which model is more reliable")
        
        # SOLUTION TEMPLATE (uncomment and complete):

        from sklearn.model_selection import cross_val_score
        
        # Prepare data
        X = self.predictor.data[self.predictor.feature_names]
        y = self.predictor.data['mpg']
        
        # Scale features for Linear Regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Models to compare
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        print("Cross-Validation Results (5-fold):")
        print("-" * 40)
        
        for name, model in models.items():
            if name == 'Linear Regression':
                scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            else:
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            mean_score = scores.mean()
            std_score = scores.std()
            
            print(f"{name}:")
            print(f"  R² scores: {scores}")
            print(f"  Mean: {mean_score:.3f} ± {std_score:.3f}")
            print()

        
    def exercise_5_new_algorithm(self):
        """
        EXERCISE 5: Try a New Algorithm
        
        Goal: Experiment with Support Vector Regression (SVR).
        
        Tasks:
        1. Train an SVR model
        2. Compare with existing models
        3. Analyze strengths and weaknesses
        """
        print("="*60)
        print("EXERCISE 5: SUPPORT VECTOR REGRESSION")
        print("="*60)
        
        print("\n🎯 YOUR TASK:")
        print("1. Import and train an SVR model")
        print("2. Use RBF kernel with default parameters")
        print("3. Compare performance with Linear Regression and Random Forest")
        print("4. Discuss when you might choose each algorithm")
        
        # SOLUTION TEMPLATE (uncomment and complete):

        from sklearn.svm import SVR
        
        # Prepare scaled data (SVR needs feature scaling)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.predictor.X_train)
        X_test_scaled = scaler.transform(self.predictor.X_test)
        
        # Train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        results = {}
        
        for name, model in models.items():
            if name in ['Linear Regression', 'SVR (RBF)']:
                # Use scaled data
                model.fit(X_train_scaled, self.predictor.y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # Random Forest doesn't need scaling
                model.fit(self.predictor.X_train, self.predictor.y_train)
                y_pred = model.predict(self.predictor.X_test)
            
            r2 = r2_score(self.predictor.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.predictor.y_test, y_pred))
            
            results[name] = {'R²': r2, 'RMSE': rmse}
            
            print(f"{name}:")
            print(f"  R² Score: {r2:.3f}")
            print(f"  RMSE: {rmse:.2f}")
            print()
        
        # Create comparison plot
        model_names = list(results.keys())
        r2_scores = [results[name]['R²'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, r2_scores, color=['blue', 'green', 'red'])
        plt.ylabel('R² Score')
        plt.title('Model Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Discussion points
        print("DISCUSSION:")
        print("- Linear Regression: Fast, interpretable, assumes linear relationships")
        print("- Random Forest: Handles non-linear patterns, robust, less interpretable")
        print("- SVR: Good for high-dimensional data, can capture complex patterns")

        
    def challenge_exercise_ensemble(self):
        """
        CHALLENGE EXERCISE: Create an Ensemble Model
        
        Goal: Combine multiple models for better performance.
        
        Tasks:
        1. Create a simple voting ensemble
        2. Use weighted averaging
        3. Compare with individual models
        """
        print("="*60)
        print("CHALLENGE: ENSEMBLE MODELING")
        print("="*60)
        
        print("\n🎯 YOUR CHALLENGE:")
        print("1. Train Linear Regression, Random Forest, and SVR")
        print("2. Create ensemble predictions using simple averaging")
        print("3. Try weighted averaging (give more weight to better models)")
        print("4. Compare ensemble performance with individual models")
        
        print("\n💡 HINTS:")
        print("- Ensemble prediction = (pred1 + pred2 + pred3) / 3")
        print("- Weighted ensemble = w1*pred1 + w2*pred2 + w3*pred3")
        print("- Weights should sum to 1.0")
        
        # SOLUTION TEMPLATE (uncomment and complete):

        # Train individual models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.predictor.X_train)
        X_test_scaled = scaler.transform(self.predictor.X_test)
        
        # Models
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        # Train models
        lr_model.fit(X_train_scaled, self.predictor.y_train)
        rf_model.fit(self.predictor.X_train, self.predictor.y_train)
        svr_model.fit(X_train_scaled, self.predictor.y_train)
        
        # Individual predictions
        lr_pred = lr_model.predict(X_test_scaled)
        rf_pred = rf_model.predict(self.predictor.X_test)
        svr_pred = svr_model.predict(X_test_scaled)
        
        # Simple ensemble (equal weights)
        ensemble_simple = (lr_pred + rf_pred + svr_pred) / 3
        
        # Weighted ensemble (based on individual R² scores)
        lr_r2 = r2_score(self.predictor.y_test, lr_pred)
        rf_r2 = r2_score(self.predictor.y_test, rf_pred)
        svr_r2 = r2_score(self.predictor.y_test, svr_pred)
        
        # Normalize weights
        total_r2 = lr_r2 + rf_r2 + svr_r2
        w_lr = lr_r2 / total_r2
        w_rf = rf_r2 / total_r2
        w_svr = svr_r2 / total_r2
        
        ensemble_weighted = w_lr * lr_pred + w_rf * rf_pred + w_svr * svr_pred
        
        # Compare results
        models_results = {
            'Linear Regression': lr_r2,
            'Random Forest': rf_r2,
            'SVR': svr_r2,
            'Simple Ensemble': r2_score(self.predictor.y_test, ensemble_simple),
            'Weighted Ensemble': r2_score(self.predictor.y_test, ensemble_weighted)
        }
        
        print("Model Performance Comparison:")
        print("-" * 30)
        for name, r2 in models_results.items():
            print(f"{name}: R² = {r2:.3f}")
        
        # Visualization
        plt.figure(figsize=(12, 6))
        names = list(models_results.keys())
        scores = list(models_results.values())
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        plt.bar(names, scores, color=colors)
        plt.ylabel('R² Score')
        plt.title('Individual Models vs Ensemble Methods')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print(f"\nWeights used in weighted ensemble:")
        print(f"Linear Regression: {w_lr:.3f}")
        print(f"Random Forest: {w_rf:.3f}")
        print(f"SVR: {w_svr:.3f}")


def run_all_exercises():
    """
    Run all exercises in sequence.
    """
    print("🎓 FUEL EFFICIENCY PREDICTOR - STUDENT EXERCISES")
    print("=" * 60)
    print("Complete these exercises to master machine learning concepts!")
    print()
    
    exercises = StudentExercises()
    
    while True:
        print("\nAvailable Exercises:")
        print("1. Feature Importance Analysis")
        print("2. Hyperparameter Tuning")
        print("3. Feature Engineering")
        print("4. Cross-Validation")
        print("5. New Algorithm (SVR)")
        print("6. Challenge: Ensemble Modeling")
        print("0. Exit")
        
        choice = input("\nSelect an exercise (0-6): ").strip()
        
        if choice == '0':
            print("Happy learning! 🎉")
            break
        elif choice == '1':
            exercises.exercise_1_feature_importance()
        elif choice == '2':
            exercises.exercise_2_hyperparameter_tuning()
        elif choice == '3':
            exercises.exercise_3_feature_engineering()
        elif choice == '4':
            exercises.exercise_4_cross_validation()
        elif choice == '5':
            exercises.exercise_5_new_algorithm()
        elif choice == '6':
            exercises.challenge_exercise_ensemble()
        else:
            print("Invalid choice. Please select 0-6.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    run_all_exercises() 