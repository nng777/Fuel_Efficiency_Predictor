"""
Quick Demo of Fuel Efficiency Predictor
=======================================

This script demonstrates the key features of the fuel efficiency predictor
for AI Engineering students.
"""

from fuel_efficiency_predictor import FuelEfficiencyPredictor
import matplotlib.pyplot as plt

def quick_demo():
    """
    Run a quick demonstration of the fuel efficiency predictor.
    """
    print("🚗 FUEL EFFICIENCY PREDICTOR - QUICK DEMO")
    print("=" * 50)
    
    # Initialize predictor
    predictor = FuelEfficiencyPredictor()
    
    # Load and explore data
    print("\n1. Loading synthetic car dataset...")
    predictor.load_data()
    
    print("\n2. Quick data overview:")
    print(f"   - Dataset size: {len(predictor.data)} cars")
    print(f"   - Features: {len(predictor.data.columns) - 1}")
    print(f"   - Average MPG: {predictor.data['mpg'].mean():.1f}")
    print(f"   - MPG range: {predictor.data['mpg'].min():.1f} - {predictor.data['mpg'].max():.1f}")
    
    # Prepare data and train models
    print("\n3. Preparing data and training models...")
    predictor.prepare_data()
    predictor.train_models()
    
    # Evaluate models
    print("\n4. Evaluating model performance...")
    results = predictor.evaluate_models()
    
    # Show best model
    best_model = max(results.keys(), key=lambda k: results[k]['R²'])
    print(f"\n🏆 Best performing model: {best_model}")
    print(f"   R² Score: {results[best_model]['R²']:.3f}")
    print(f"   RMSE: {results[best_model]['RMSE']:.2f} MPG")
    
    # Demo predictions
    print("\n5. Making predictions for sample cars...")
    
    sample_cars = [
        {
            'name': 'Efficient 4-cylinder',
            'cylinders': 4,
            'displacement': 120,
            'horsepower': 85,
            'weight': 2200,
            'acceleration': 18,
            'model_year': 80
        },
        {
            'name': 'Powerful 8-cylinder',
            'cylinders': 8,
            'displacement': 350,
            'horsepower': 200,
            'weight': 3500,
            'acceleration': 12,
            'model_year': 75
        },
        {
            'name': 'Balanced 6-cylinder',
            'cylinders': 6,
            'displacement': 200,
            'horsepower': 130,
            'weight': 2800,
            'acceleration': 15,
            'model_year': 78
        }
    ]
    
    for car in sample_cars:
        car_features = {k: v for k, v in car.items() if k != 'name'}
        print(f"\n   {car['name']}:")
        
        # Make predictions with both models
        import pandas as pd
        car_df = pd.DataFrame([car_features])
        
        # Linear Regression prediction
        car_scaled = predictor.scaler.transform(car_df)
        lr_pred = predictor.models['Linear Regression'].predict(car_scaled)[0]
        
        # Random Forest prediction
        rf_pred = predictor.models['Random Forest'].predict(car_df)[0]
        
        print(f"     Linear Regression: {lr_pred:.1f} MPG")
        print(f"     Random Forest: {rf_pred:.1f} MPG")
        print(f"     Average: {(lr_pred + rf_pred) / 2:.1f} MPG")
    
    # Feature importance
    print("\n6. Most important features (Random Forest):")
    rf_model = predictor.models['Random Forest']
    importance = rf_model.feature_importances_
    
    feature_importance = list(zip(predictor.feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, imp) in enumerate(feature_importance[:3]):
        print(f"   {i+1}. {feature}: {imp:.3f}")
    
    print("\n✅ Demo complete!")
    print("\nNext steps:")
    print("- Run 'python fuel_efficiency_predictor.py' for full analysis")
    print("- Run 'streamlit run streamlit_app.py' for interactive web app")
    print("- Run 'python student_exercises.py' for hands-on practice")

if __name__ == "__main__":
    quick_demo() 