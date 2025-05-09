# Food Classifier

This repository contains a deep learning-based food classifier that distinguishes between food and non-food images using a fine-tuned GoogLeNet model. The project includes a training notebook, a Flask server for inference, and the Food-5K dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Flask Server](#running-the-flask-server)
  - [Making Predictions](#making-predictions)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The food classifier is built using PyTorch and leverages a pre-trained GoogLeNet model, fine-tuned on the Food-5K dataset. The model classifies images as either "Food" or "Non-Food" with high accuracy. The repository includes a Jupyter notebook for training and evaluation, a Flask API for serving predictions, and scripts to visualize results such as confusion matrices and ROC curves.

## Dataset
The Food-5K dataset is included in the repository under the `Food-5K` directory. It contains images labeled as follows:
- **Food**: Images starting with `1_xxx.jpg` (label = 1)
- **Non-Food**: Images starting with `0_xxx.jpg` (label = 0)

The dataset is split into:
- **Training**: `Food-5K/training`
- **Validation**: `Food-5K/validation`
- **Evaluation**: `Food-5K/evaluation`

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- PIL (Pillow)
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- TQDM
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/food-classifier.git
   cd food-classifier
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the Food-5K dataset is in the `Food-5K` directory.

## Usage

### Training the Model
1. Open the `food_classifier_notebook.ipynb` in Jupyter Notebook or JupyterLab.
2. Run the cells sequentially to:
   - Load and preprocess the Food-5K dataset.
   - Fine-tune the GoogLeNet model.
   - Train the model for 10 epochs.
   - Evaluate performance on the test set.
   - Save the best model as `googlenet_food_best.pth`.
3. The notebook also generates visualizations (e.g., sample images, confusion matrix, ROC curve).

Example command to start Jupyter:
```bash
jupyter notebook
```

### Running the Flask Server
1. Ensure the trained model (`googlenet_food_best.pth`) is in the root directory.
2. Run the Flask server:
   ```bash
   python server.py
   ```
3. The server will start on `http://0.0.0.0:5001`.

### Making Predictions
- **Via Notebook**: Use the `predict_single_image` function in the notebook to classify a single image. Example:
  ```python
  result, confidence = predict_single_image("googlenet_food_best.pth", "Food-5K/evaluation/0_54.jpg")
  ```

- **Via Flask API**: Send a POST request to the `/predict` endpoint with an image file:
  ```bash
  curl -X POST -F "image=@path/to/image.jpg" http://localhost:5001/predict
  ```
  Response example:
  ```json
  {
    "prediction": "Food",
    "confidence": "95.32%"
  }
  ```

## Project Structure
```
food-classifier/
├── Food-5K/
│   ├── training/
│   ├── validation/
│   ├── evaluation/
├── food_classifier_notebook.ipynb  # Training and evaluation notebook
├── server.py                      # Flask API server
├── googlenet_food_best.pth        # Trained model (generated after training)
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Results
The model achieves high accuracy on the Food-5K dataset. Key metrics (based on the test set):
- **Accuracy**: ~95% (varies by run)
- **Precision, Recall, F1**: Detailed in the notebook's classification report
- **AUC Score**: Visualized in the ROC curve

Sample visualizations (confusion matrix, ROC curve) are generated in the notebook.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
