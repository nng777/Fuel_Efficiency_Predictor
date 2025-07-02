import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FuelEfficiencyPredictor:
    """
    A comprehensive fuel efficiency prediction system for AI Engineering students.
    
    This class demonstrates:
    - Data loading and exploration
    - Data preprocessing
    - Multiple ML models (Linear Regression, Random Forest)
    - Model evaluation and comparison
    - Making predictions on new data
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = 'mpg'
        
    def create_sample_data(self, n_samples=1000):
        """
        Create a realistic synthetic dataset for fuel efficiency prediction.
        Features include: cylinders, displacement, horsepower, weight, acceleration, model_year
        """
        np.random.seed(42)
        
        # Generate realistic car features
        cylinders = np.random.choice([4, 6, 8], n_samples, p=[0.6, 0.3, 0.1])
        displacement = np.random.normal(200, 50, n_samples)
        displacement = np.clip(displacement, 100, 400)
        
        horsepower = np.random.normal(120, 30, n_samples)
        horsepower = np.clip(horsepower, 60, 250)
        
        weight = np.random.normal(2800, 400, n_samples)
        weight = np.clip(weight, 1800, 4500)
        
        acceleration = np.random.normal(15, 3, n_samples)
        acceleration = np.clip(acceleration, 8, 25)
        
        model_year = np.random.randint(70, 82, n_samples)
        
        # Calculate MPG based on realistic relationships
        # Higher displacement, horsepower, weight -> lower MPG
        # Higher acceleration, newer model year -> higher MPG
        mpg = (
            40 
            - 0.05 * displacement 
            - 0.08 * horsepower 
            - 0.003 * weight 
            + 0.5 * acceleration 
            + 0.3 * (model_year - 70)
            - 2 * (cylinders - 4)
            + np.random.normal(0, 2, n_samples)  # Add some noise
        )
        mpg = np.clip(mpg, 10, 45)  # Realistic MPG range
        
        # Create DataFrame
        data = pd.DataFrame({
            'cylinders': cylinders,
            'displacement': displacement,
            'horsepower': horsepower,
            'weight': weight,
            'acceleration': acceleration,
            'model_year': model_year,
            'mpg': mpg
        })
        
        return data
    
    def load_data(self, data_path=None):
        """
        Load the dataset. If no path provided, create synthetic data.
        """
        if data_path is None:
            print("Creating synthetic fuel efficiency dataset...")
            self.data = self.create_sample_data()
        else:
            print(f"Loading data from {data_path}...")
            self.data = pd.read_csv(data_path)
        
        print(f"Dataset loaded with {len(self.data)} samples and {len(self.data.columns)} features")
        return self.data
    
    def explore_data(self):
        """
        Perform exploratory data analysis to understand the dataset.
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print("\nDataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns)}")
        
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        # Correlation analysis
        print("\nCorrelation with MPG:")
        correlations = self.data.corr()['mpg'].sort_values(ascending=False)
        print(correlations)
        
        return correlations
    
    def visualize_data(self):
        """
        Create visualizations to understand the data better.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fuel Efficiency Dataset Analysis', fontsize=16)
        
        # Distribution of target variable
        axes[0, 0].hist(self.data['mpg'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution of MPG')
        axes[0, 0].set_xlabel('Miles Per Gallon')
        axes[0, 0].set_ylabel('Frequency')
        
        # Correlation heatmap
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Feature Correlation Matrix')
        
        # Scatter plots for key relationships
        axes[0, 2].scatter(self.data['weight'], self.data['mpg'], alpha=0.6)
        axes[0, 2].set_title('Weight vs MPG')
        axes[0, 2].set_xlabel('Weight')
        axes[0, 2].set_ylabel('MPG')
        
        axes[1, 0].scatter(self.data['horsepower'], self.data['mpg'], alpha=0.6)
        axes[1, 0].set_title('Horsepower vs MPG')
        axes[1, 0].set_xlabel('Horsepower')
        axes[1, 0].set_ylabel('MPG')
        
        axes[1, 1].scatter(self.data['displacement'], self.data['mpg'], alpha=0.6)
        axes[1, 1].set_title('Displacement vs MPG')
        axes[1, 1].set_xlabel('Displacement')
        axes[1, 1].set_ylabel('MPG')
        
        # Box plot for cylinders
        self.data.boxplot(column='mpg', by='cylinders', ax=axes[1, 2])
        axes[1, 2].set_title('MPG by Number of Cylinders')
        axes[1, 2].set_xlabel('Cylinders')
        axes[1, 2].set_ylabel('MPG')
        
        plt.tight_layout()
        plt.show()
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare the data for machine learning by splitting and scaling.
        """
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        
        # Separate features and target
        self.feature_names = [col for col in self.data.columns if col != self.target_name]
        X = self.data[self.feature_names]
        y = self.data[self.target_name]
        
        print(f"Features: {self.feature_names}")
        print(f"Target: {self.target_name}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Features scaled using StandardScaler")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple machine learning models for comparison.
        """
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Linear Regression
        print("Training Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(self.X_train_scaled, self.y_train)
        self.models['Linear Regression'] = lr_model
        
        # Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)  # Random Forest doesn't need scaling
        self.models['Random Forest'] = rf_model
        
        print("Models trained successfully!")
        
        return self.models
    
    def evaluate_models(self):
        """
        Evaluate all trained models and compare their performance.
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Make predictions
            if name == 'Linear Regression':
                y_pred = model.predict(self.X_test_scaled)
            else:
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'Predictions': y_pred
            }
            
            print(f"  Mean Squared Error: {mse:.2f}")
            print(f"  Root Mean Squared Error: {rmse:.2f}")
            print(f"  Mean Absolute Error: {mae:.2f}")
            print(f"  R² Score: {r2:.3f}")
        
        self.results = results
        return results
    
    def visualize_predictions(self):
        """
        Create visualizations to compare model predictions.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Model Predictions vs Actual Values', fontsize=16)
        
        colors = ['blue', 'red']
        
        for i, (name, result) in enumerate(self.results.items()):
            y_pred = result['Predictions']
            
            axes[i].scatter(self.y_test, y_pred, alpha=0.6, color=colors[i])
            axes[i].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 
                        'k--', lw=2)
            axes[i].set_xlabel('Actual MPG')
            axes[i].set_ylabel('Predicted MPG')
            axes[i].set_title(f'{name}\nR² = {result["R²"]:.3f}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance(self):
        """
        Display feature importance for models that support it.
        """
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        
        # Random Forest feature importance
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            importance = rf_model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            print("\nRandom Forest Feature Importance:")
            print(feature_importance_df)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
            plt.xlabel('Importance')
            plt.title('Feature Importance (Random Forest)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        
        # Linear Regression coefficients
        if 'Linear Regression' in self.models:
            lr_model = self.models['Linear Regression']
            coefficients = lr_model.coef_
            
            coef_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': coefficients
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            print("\nLinear Regression Coefficients:")
            print(coef_df)
    
    def predict_new_car(self, car_features):
        """
        Make predictions for a new car using all trained models.
        
        Args:
            car_features (dict): Dictionary with feature names and values
        """
        print("\n" + "="*50)
        print("PREDICTION FOR NEW CAR")
        print("="*50)
        
        # Convert to DataFrame
        new_car_df = pd.DataFrame([car_features])
        
        print("Car specifications:")
        for feature, value in car_features.items():
            print(f"  {feature}: {value}")
        
        print("\nPredictions:")
        
        for name, model in self.models.items():
            if name == 'Linear Regression':
                # Scale the features for linear regression
                new_car_scaled = self.scaler.transform(new_car_df)
                prediction = model.predict(new_car_scaled)[0]
            else:
                prediction = model.predict(new_car_df)[0]
            
            print(f"  {name}: {prediction:.2f} MPG")
    
    def run_complete_analysis(self):
        """
        Run the complete machine learning pipeline.
        """
        print("FUEL EFFICIENCY PREDICTION - COMPLETE ANALYSIS")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Visualize data
        self.visualize_data()
        
        # Step 4: Prepare data
        self.prepare_data()
        
        # Step 5: Train models
        self.train_models()
        
        # Step 6: Evaluate models
        self.evaluate_models()
        
        # Step 7: Visualize predictions
        self.visualize_predictions()
        
        # Step 8: Feature importance
        self.feature_importance()
        
        # Step 9: Example prediction
        example_car = {
            'cylinders': 4,
            'displacement': 150,
            'horsepower': 100,
            'weight': 2500,
            'acceleration': 16,
            'model_year': 78
        }
        self.predict_new_car(example_car)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)

def main():
    """
    Main function to demonstrate the fuel efficiency predictor.
    """
    # Create and run the predictor
    predictor = FuelEfficiencyPredictor()
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main() 