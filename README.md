# Assignment_01

# House Prices Prediction with PyTorch

## Methodology / Assignment Description

This assignment demonstrates an end-to-end workflow for predicting house sale prices using a neural network built in PyTorch. Our methodology involves:

- **Feature Selection and Preprocessing**:  
  We select a subset of relevant numeric features from the dataset and handle missing data by dropping incomplete records. The features are then normalized using MinMaxScaler.

- **Train/Validation Split**:  
  The dataset from `train.csv` is split into an 80% training set and a 20% validation set. This internal split allows us to monitor the model's performance and prevent overfitting.

- **Neural Network Modeling**:  
  We build a Multi-Layer Perceptron (MLP) with two hidden layers using ReLU activations. This model is designed for a regression task to predict house sale prices.

- **Hyperparameter Tuning**:  
  A simple grid search is implemented to test different hyperparameter settings (such as hidden layer sizes and learning rates). The best configuration is selected based on validation loss.

- **Model Training and Evaluation**:  
  The model is retrained using the best hyperparameters over a fixed number of epochs. Training and validation losses are recorded and plotted to evaluate model performance using metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

- **Test Data Inference**:  
  Finally, the trained model is used to generate predictions on the test dataset (`test.csv`), which lacks the target sale prices.

## Using Google Colab

1. **Upload Files**: Upload the notebook (`notebook.ipynb`) and datasets (`train.csv` and `test.csv`) to your Google Drive.
2. **Open in Colab**: Open the notebook in Google Colab.
3. **Run Cells Sequentially**: Execute the cells in order to perform data preprocessing, training, hyperparameter tuning, evaluation, and prediction.
4. **Visualize Results**: The notebook will generate plots and print evaluation metrics (MSE, RMSE) to help you understand model performance.
