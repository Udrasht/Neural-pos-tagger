# Neural-POS-Tagger
### Explanation:
- **2022201020_assignment2_report.pdf**: This file contains the report for the assignment.
- **INLP_Assignment.pdf**: This file contains problem statement.
- **README.md**: This file provides instructions and explanations for running the code.
- **UD_English-Atis**: This directory contains the dataset files in CoNLL-U format.
- **pos_tagger.py**: The main Python script implementing the POS tagger using FFNN and LSTM models.
- **trained_model**: This folder contain .pt file for lstm and ffnn model.
- **requirements.txt**: File listing the Python dependencies required to run the code.
- **Notebook_files**: folder contain Notebook(.ipynb) file for both the models(FFNN, LSTM(RNN)) with the help of these file i create pos_tagger.py file .

## Requirements
- Python 3
- PyTorch
- scikit-learn
- pandas
- seaborn

## Usage
1. Install the required dependencies:
    ```
    $ pip install -r requirements.txt
    ```

2. Execute the `pos_tagger.py` script:
    ```
    $ python pos_tagger.py <model_type>
    ```
   Replace `<model_type>` with `-f` for FFNN or `-r` for LSTM.
   ### Example:
- To run the POS tagger with the Feed Forward Neural Network (FFNN) model, use the following command:
    ```
    $ python pos_tagger.py -f
    ```
  This command will prompt you to enter a sentence for POS tagging.

- To run the POS tagger with the Recurrent Neural Network (RNN) model, use the following command:
    ```
    $ python pos_tagger.py -r
    ```
  This command will prompt you to enter a sentence for POS tagging.

3. Follow the prompts to input sentences for POS tagging. Press `q` exit the program.

## Notes
- Ensure that the dataset files and the saved model file (`lstm_model1.pt`,`ffnn_model_best.pt`) are placed in the specified directories as mentioned in the directory structure.
- The model type (`-f` for FFNN, `-r` for LSTM) must be specified as a command-line argument.
- The paths to the dataset files and the saved model file are hardcoded in the script and should be adjusted if necessary.
