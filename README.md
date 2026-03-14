# 🌿 Crop Disease Prediction using Deep Learning

A deep learning based web application that detects plant diseases from leaf images.

The system allows a user to upload an image of a plant leaf and predicts the disease affecting the crop using a trained deep learning model.

---

# 🚀 Features

* Upload plant leaf images
* Predict plant disease automatically
* Web based interface
* Deep Learning model for classification
* Fast prediction using trained PyTorch model

---

# 🧠 Model Details

The model is trained using the **PlantVillage Dataset**.

The deep learning model learns patterns from leaf images and classifies them into different disease categories.

Framework used:

* PyTorch
* Python

---

# 🌱 Supported Crops and Diseases

The model can detect diseases for the following crops:

### 🍎 Apple

* Apple Scab
* Black Rot
* Cedar Apple Rust
* Healthy

### 🍒 Cherry

* Powdery Mildew
* Healthy

### 🌽 Corn (Maize)

* Cercospora Leaf Spot
* Common Rust
* Northern Leaf Blight
* Healthy

### 🍇 Grape

* Black Rot
* Esca (Black Measles)
* Leaf Blight
* Healthy

### 🍑 Peach

* Bacterial Spot
* Healthy

### 🌶 Pepper (Bell)

* Bacterial Spot
* Healthy

### 🥔 Potato

* Early Blight
* Late Blight
* Healthy

### 🍓 Strawberry

* Leaf Scorch
* Healthy

---

# 🗂 Project Structure

```
CROP DIS
│
├── src
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│
├── static
│   ├── index.html
│   ├── style.css
│   ├── script.js
│
├── app.py
├── best_model.pth
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```
git clone https://github.com/AdarshRaj-404/Crop_disease_prediction.git
```

Move into the project folder

```
cd Crop_disease_prediction
```

Install dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the application:

```
python app.py
```

Open browser and go to:

```
http://localhost:5000
```

Upload a plant leaf image and the model will predict the disease.

---

# 📊 Dataset

This project uses the **PlantVillage Dataset**.

Dataset contains thousands of labeled images of healthy and diseased plant leaves.

Dataset is not included in this repository due to large size.

---

# 💡 Future Improvements

* Deploy model online
* Improve accuracy with larger datasets
* Add mobile support
* Add more crop diseases

---

# ⭐ If you like this project

Please consider giving this repository a **star ⭐ on GitHub**.
