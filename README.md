Sure! Here's your entire `README.md` content wrapped inside a **single large code block** (useful if you want to copy-paste it somewhere as a plain text block):

<pre lang="markdown">
```markdown
# 🍔 Food Classifier using PyTorch and GoogLeNet

This project classifies images as **Food** or **Non-Food** using a fine-tuned GoogLeNet (Inception v1) model. It uses the [Food-5K dataset](https://www.kaggle.com/datasets/food/Food-5K) and provides a Flask API for prediction.

---

## 📁 Project Structure

```
Food-5K/                   # Dataset (train/validation/eval folders)
food_classifier.ipynb     # Model training and evaluation notebook
googlenet_food_best.pth   # Saved PyTorch model
app.py                    # Flask API for inference
README.md                 # Project documentation
```

---

## 🚀 Features

- GoogLeNet (Inception v1) pretrained model
- Binary classification: Food / Non-Food
- Training, validation, and evaluation using PyTorch
- Evaluation metrics: Accuracy, F1, ROC-AUC
- Flask API for real-time predictions

---

## 📊 Dataset Format

- **Food images:** start with `1_`  
- **Non-food images:** start with `0_`  
- Folders: `training`, `validation`, `evaluation`

---

## 🧠 Model Details

- Base model: `torchvision.models.googlenet(pretrained=True)`
- Last fully connected layer changed to 2 output classes
- Early layers frozen for transfer learning

---

## 🧪 How to Train

Open `food_classifier.ipynb` in Jupyter Notebook.

```python
train_model(model, train_loader, val_loader, epochs=10)
```

After training, the model is saved as:
```
googlenet_food_best.pth
```

---

## 📦 Flask API

### 🔧 Run the API

```bash
pip install flask torch torchvision pillow
python app.py
```

### 📤 API Usage

**POST** `/predict`  
**Form-data:**  
- `image`: Upload image file

**Response:**
```json
{
  "prediction": "Food",
  "confidence": 97.32
}
```

---

## 🔍 Example Prediction

```python
predict_single_image("googlenet_food_best.pth", "Food-5K/evaluation/0_54.jpg")
```

Output:
```
Predicted Class: Non-Food | Confidence: 89.12%
```

---

## ✅ Requirements

- Python 3.7+
- PyTorch
- torchvision
- Flask
- Pillow
- scikit-learn
- matplotlib
- tqdm
- seaborn

Install via:

```bash
pip install -r requirements.txt
```

---

## 👤 Author

**Tayyab Anees**  
FAST NUCES, Lahore  
Final Year Project

---

## 📄 License

MIT License
```
</pre>

Let me know if you want this saved as a `.md` file too.
