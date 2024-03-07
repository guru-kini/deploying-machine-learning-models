import model.data.data_helper as dataHelper
from model.pipeline import titanic_pipeline
from sklearn.model_selection import train_test_split
from model.config.configuration import _config as config
from model.persistence import save_pipeline

def run_training() -> None:
    # Load the Data
    data = dataHelper.load_data(file_name="titanic-raw.csv")
    
    # Split into training and test data
    X_train, X_test, \
        y_train, y_test = train_test_split(
                            data.drop('survived', axis=1),  # predictors
                            data[config.model_config.target],  # target
                            test_size=config.model_config.test_size,  # percentage of obs in test set
                            random_state=config.app_config.seed)  # seed to ensure reproducibility
    print("Training Data Shape: " + str(X_train.shape))
    print("Test Data Shape: " + str(X_test.shape))
    
    # Fit the model
    titanic_pipeline.fit(X_train, y_train)
    
    # Save the trained model
    save_pipeline(pipeline_to_persist=titanic_pipeline)
    
if __name__ == "__main__":
    run_training()
