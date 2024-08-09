Text Classification with Neural Networks in TensorFlow
This repository contains various neural network models implemented using TensorFlow for text classification tasks. The models explored include GRU with Attention, hyperparameter optimization using Optuna, and a Feed-Forward Neural Network (FFNN). The project aims to compare the performance of these models on text data using validation loss and accuracy as evaluation metrics.

Project Structure
GRU with Attention Model:

Architecture: Utilizes a GRU layer followed by an Attention mechanism to capture the importance of words in sequences.
Components: Embedding, GRU, Attention, GlobalAveragePooling1D, Dense, and Dropout layers.
Performance: The model achieved a validation loss of 0.3057 and validation accuracy of 0.87.
Hyperparameter Optimization with Optuna:

Objective: To find the optimal hyperparameters (units, dropout rate, learning rate) for a GRU-based model.
Methodology: Conducts multiple trials with different hyperparameter combinations and evaluates model performance on validation data.
Results: The best model achieved a validation accuracy of 0.5045 with a corresponding validation loss of 3.0902.
Feed-Forward Neural Network (FFNN):

Architecture: A simple neural network with an Embedding layer, followed by a Flatten layer, Dense layers, and Dropout for regularization.
Components: Embedding, Flatten, Dense, and Dropout layers.
Performance: The FFNN model recorded a validation loss of 0.6967 and validation accuracy of 0.5035.
Installation
To run the models in this project, you need to have Python installed along with TensorFlow, Optuna, and NumPy.

Install the required packages:

bash
Copy code
pip install tensorflow optuna numpy
Usage
To run the models:

GRU with Attention:

Navigate to the script defining the GRU model with Attention and execute the file.
The model will train and output validation loss and accuracy.
Optuna Hyperparameter Tuning:

Run the Optuna script to start the hyperparameter optimization process.
The script will output the best hyperparameters and corresponding validation performance.
Feed-Forward Neural Network (FFNN):

Execute the FFNN script to train the model and evaluate its performance on the validation set.
Analysis
GRU with Attention: This model performed the best in terms of validation accuracy, indicating that the combination of GRU and Attention can effectively capture relevant features in sequential text data.

Optuna Tuning: Although Optuna found the best set of hyperparameters, the resulting model's performance suggests that further fine-tuning or adjustments might be needed to improve its effectiveness.

FFNN: The FFNN model's simpler architecture led to lower performance compared to the GRU with Attention model, highlighting the importance of using more complex architectures for text classification tasks.

Conclusion
This project demonstrates the application of different neural network architectures and hyperparameter optimization techniques for text classification. The results indicate that more complex models, such as GRU with Attention, can offer better performance in handling sequential data compared to simpler models like FFNN.
