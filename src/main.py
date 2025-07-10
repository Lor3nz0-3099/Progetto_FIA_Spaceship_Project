from preliminar_operations import PreliminarOperations
from model_factory import ModelFactory
from testing import Testing
import joblib
import os


if __name__ == "__main__":

    # Initialized PreliminarOperations class
    preliminar_ops = PreliminarOperations()
    
    # Load and prepare raw dataset for the new pipeline
    X_train, X_val, y_train, y_val = preliminar_ops.get_raw_data_for_pipeline()
    
    # Print the shapes of the training and validation sets
    print(f"Raw Train shape: {X_train.shape}, Raw Val shape: {X_val.shape}")
    print("Note: Feature engineering and logical imputations will be applied during training")

    # Choose the model to train
    print("\nWhich model do you want to train/use?\n1: Random Forest\n2: AdaBoost\n3: SVM\n4: XGBoost\n5: CatBoost\n6: Stacked Classifier\nEnter choice (1/2/3/4/5/6): ", end="")
    choice = input().strip()
    
    model_dictionary = {
        '1': "RandomForest_model_new_pipeline.joblib",
        '2': "AdaBoost_model_new_pipeline.joblib",
        '3': "SVM_model_new_pipeline.joblib",
        '4': "XGBoost_model_new_pipeline.joblib",
        '5': "CatBoost_model_new_pipeline.joblib",
        '6': "StackedClassifier_model_new_pipeline.joblib"
    }
    
    model_filename = model_dictionary.get(choice)
    model_path = f"models/{model_filename}"
    
    # Check if the model file already exists
    if os.path.exists(model_path):
        print(f"Model {model_filename} already exists. Loading the model...")
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        # Skip training and evaluation if the model already exists
        save_choice = 'y'
        model_name = model_filename
        model_type = model_filename.split('_')[0]
    else:
        print("Training the model with new pipeline (includes logical imputations)...")
        model_factory = ModelFactory()
        model = model_factory.train_and_evaluate_model(choice, X_train, y_train, X_val, y_val)
        
        # Save the trained model
        save_choice = input("Do you want to save this model? (y/n): ").strip().lower()
        if save_choice == 'y':
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
            model_name = model_filename
            model_type = model_filename.split('_')[0]
        else:
            model_name = None
    
    # Test the model if requested
    if model_name:
        test_choice = input("Do you want to test the model on test data? (y/n): ").strip().lower()
        if test_choice == 'y':
            testing = Testing()
            testing.test_model(model, model_type)
