# Import necessary modules for the machine learning pipeline
from preliminar_operations import PreliminarOperations  
from model_factory import ModelFactory                 
from testing import Testing                             
import joblib                                          
import os                                              


if __name__ == "__main__":
    
    # ===== DATA PREPARATION PHASE =====
    print("=== SPACESHIP PROJECT - ML PIPELINE ===\n")
    
    # Initialize the preliminary operations handler
    # This class manages data loading, cleaning, and initial preprocessing
    preliminar_ops = PreliminarOperations()
    
    # Load and prepare the raw dataset for the machine learning pipeline
    # Returns training and validation sets (features and targets)
    # Note: Raw data will be further processed during model training
    X_train, X_val, y_train, y_val = preliminar_ops.get_raw_data_for_pipeline()
    
    # Display dataset information for user awareness
    print(f"Raw Train shape: {X_train.shape}, Raw Val shape: {X_val.shape}")
    print("Note: Feature engineering and logical imputations will be applied during training\n")

    # ===== MODEL SELECTION PHASE =====
    # Present available machine learning models to the user
    print("Which model do you want to train/use?")
    print("1: Random Forest")      # Ensemble method, good for general classification
    print("2: AdaBoost")          # Adaptive boosting algorithm
    print("3: SVM")               # Support Vector Machine
    print("4: XGBoost")           # Gradient boosting framework
    print("5: CatBoost")          # Categorical boosting algorithm
    print("6: Stacked Classifier") # Ensemble of multiple models
    choice = input("Enter choice (1/2/3/4/5/6): ").strip()
    
    # Map user choice to corresponding model filename
    # Each model is saved with a specific naming convention for the new pipeline
    model_dictionary = {
        '1': "RandomForest_model_new_pipeline.joblib",
        '2': "AdaBoost_model_new_pipeline.joblib",
        '3': "SVM_model_new_pipeline.joblib",
        '4': "XGBoost_model_new_pipeline.joblib",
        '5': "CatBoost_model_new_pipeline.joblib",
        '6': "StackedClassifier_model_new_pipeline.joblib"
    }
    
    # Get the filename for the selected model
    model_filename = model_dictionary.get(choice)
    if not model_filename:
        print("Invalid choice! Please run the script again and select a valid option.")
        exit(1)
    
    # Construct the full path to the model file
    model_path = f"models/{model_filename}"
    
    # ===== MODEL LOADING OR TRAINING PHASE =====
    # Check if a pre-trained model already exists to avoid unnecessary retraining
    if os.path.exists(model_path):
        print(f"\nModel {model_filename} already exists. Loading the model...")
        
        # Load the pre-trained model from disk
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        
        # Set variables for the loaded model
        # Skip training since we're using an existing model
        save_choice = 'y'  # Assume we want to keep the existing model
        model_name = model_filename
        model_type = model_filename.split('_')[0]  # Extract model type (e.g., 'RandomForest')
        
    else:
        print(f"\nModel {model_filename} not found. Training new model...")
        print("Training the model with new pipeline (includes logical imputations)...")
        
        # Initialize the model factory for creating and training models
        model_factory = ModelFactory()
        
        # Train and evaluate the selected model
        # This includes feature engineering, logical imputations, classical preprocessing and model training
        model = model_factory.train_and_evaluate_model(choice, X_train, y_train, X_val, y_val)
        
        # ===== MODEL SAVING PHASE =====
        # Ask user if they want to save the newly trained model
        save_choice = input("\nDo you want to save this model? (y/n): ").strip().lower()
        
        if save_choice == 'y':
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Save the trained model to disk using joblib
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
            
            # Set model metadata for potential testing
            model_name = model_filename
            model_type = model_filename.split('_')[0]
        else:
            print("Model not saved.")
            model_name = None  # No model to test if not saved
    
    # ===== MODEL TESTING PHASE =====
    # Offer to test the model on test data if we have a valid model
    if model_name:
        print(f"\nModel '{model_type}' is ready for testing.")
        test_choice = input("Do you want to test the model on test data? (y/n): ").strip().lower()
        
        if test_choice == 'y':
            print("\nInitializing model testing...")
            
            # Initialize testing module and run model evaluation
            testing = Testing()
            testing.test_model(model)
            
            print("\nTesting completed!")
        else:
            print("Testing skipped.")
    else:
        print("\nNo model available for testing.")
    
    print("\n=== PIPELINE EXECUTION COMPLETED ===")
