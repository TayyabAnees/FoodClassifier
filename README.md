🍔 Food Classifier using PyTorch and GoogLeNet
This repository contains a deep learning project that classifies images into Food and Non-Food categories using the Food-5K dataset. It leverages GoogLeNet (Inception v1), trained and evaluated using PyTorch. A Flask API is also included to serve predictions.

📁 Repository Structure
bash
Copy
Edit
├── Food-5K/                   # Dataset (training/validation/evaluation folders)
├── food_classifier.ipynb     # Jupyter notebook for training and evaluation
├── googlenet_food_best.pth   # Trained model weights
├── app.py                    # Flask server to serve predictions
└── README.md                 # Project documentation
🚀 Features
Binary classification of Food vs. Non-Food

Transfer learning with GoogLeNet

Data augmentation on training images

Evaluation with metrics: Accuracy, F1-score, Confusion Matrix, ROC-AUC

Flask API for inference on custom images

🧠 Model Architecture
Pretrained GoogLeNet from torchvision.models

Final fully connected layer modified to output 2 classes

Frozen early layers for efficient fine-tuning

🗃️ Dataset
The Food-5K dataset is used with the following structure:

training/ - Training set

validation/ - Validation set

evaluation/ - Test set

Naming convention:

Images starting with 1_ → Food

Images starting with 0_ → Non-Food

🛠️ Training
Run the Jupyter notebook food_classifier.ipynb to:

Load and transform the data

Train and evaluate the GoogLeNet model

Save the best model to googlenet_food_best.pth

Sample training command (inside notebook):
python
Copy
Edit
train_model(model, train_loader, val_loader, epochs=10)
📈 Evaluation Metrics
Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1)

ROC Curve & AUC

Plots are visualized using Matplotlib and Seaborn.

📦 Flask API
A lightweight Flask server (app.py) is provided to classify a single uploaded image.

🔧 How to Run the API:
bash
Copy
Edit
pip install flask torchvision torch pillow
python app.py
📤 API Endpoint
POST /predict

Request (multipart/form-data):

image: Image file

Response (JSON):

json
Copy
Edit
{
  "prediction": "Food",
  "confidence": 97.32
}
📷 Sample Prediction
python
Copy
Edit
predict_single_image("googlenet_food_best.pth", "Food-5K/evaluation/0_54.jpg")
Output:

yaml
Copy
Edit
Predicted Class: Non-Food | Confidence: 89.12%
Image will be displayed with prediction label.

✅ Dependencies
Python 3.7+

PyTorch

torchvision

Flask

Pillow

scikit-learn

matplotlib

tqdm

seaborn

Install all dependencies:

bash
Copy
Edit
pip install -r requirements.txt
(You can generate requirements.txt using pip freeze > requirements.txt)

👨‍💻 Author
Tayyab Anees
Final Year Project – FAST NUCES Lahore
Project: Food Image Classifier – Deep Learning + Flask

📄 License
This project is licensed under the MIT License.
