from preliminar_operations import PreliminarOperations
from model_factory import ModelFactory
import joblib


if __name__ == "__main__":

    # Initialized PreliminarOperations class
    preliminar_ops = PreliminarOperations()
    
    # Load and prepare the dataset
    X_train, X_val, y_train, y_val = preliminar_ops.preliminar_operations()
    
    # Print the shapes of the training and validation sets
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Choose the model to train
    print("\n Which model do you want to train? (1: Random Forest, 2: AdaBoost): ", end="")
    choice = input().strip()
    
    # Create ModelFactory instance
    model_factory = ModelFactory()
    model = model_factory.train_and_evaluate_model(choice, X_train, y_train, X_val, y_val)

    # Save the trained model
    print("Do you want to save the model? (y/n): ", end="")
    save_choice = input().strip().lower()
    if save_choice != 'y':
        print("Model not saved.")
        exit()
    else:
        print("Saving model...")
        model_name = "RandomForest_model.joblib" if choice == '1' else "AdaBoost_model.joblib"
        joblib.dump(model, f"models/{model_name}")
        print(f"Model saved as models/{model_name}")


    
    


