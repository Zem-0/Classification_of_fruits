# 🥦 Fresh vs Rotten Fruit Classifier 🍎

A Convolutional Neural Network (CNN)-based image classification project that distinguishes between **fresh** and **rotten** fruits using TensorFlow and Keras. The model has been trained on a well-structured image dataset and achieves high accuracy in predicting the freshness of a fruit, making it valuable for food safety, logistics, and quality control applications.

## 📌 Abstract

In today's fast-paced world, ensuring the quality of perishable goods like fruits is a major concern. Manual inspection is not only time-consuming but also prone to human error. This project introduces a deep learning approach to classify fruits as either **fresh** or **rotten** using image data. Leveraging the power of Convolutional Neural Networks (CNNs), the model learns intricate patterns and color-based features that distinguish healthy produce from spoiled ones.

Our model was trained on a dataset containing categorized images of fresh and rotten fruits. Through proper preprocessing, augmentation, and training, the classifier can assist in automating quality control checks in supermarkets, warehouses, and food packaging industries.

---


## 🧠 Model Architectures

### 1. CNN from Scratch
- Built using multiple convolutional and pooling layers
- Dropout layers for regularization
- Final softmax layer for binary classification
- Custom feature extraction pipeline

### 2. VGG16 (Transfer Learning)
- Used pretrained VGG16 (excluding top layers)
- Added custom dense layers for classification
- All layers frozen except top dense layers

### 3. ResNet50 (Transfer Learning)
- Used pretrained ResNet50 base
- Fine-tuned top layers
- Added global average pooling and dense layers

---

## 📊 Model Performance Comparison

| Model           | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|----------------|-------------------|---------------------|---------------|-----------------|
| CNN (Scratch)  |  94.59%           |  94.25%             |  0.012        |  0.125          |
| VGG16          |   98%             |  97.20%             |  0.003        |  0.089          |
| ResNet50       |  70.98%           |  70.75%             |  0.005        |  0.072          |

🔍 **Insight:**  
While the custom CNN achieved excellent results, **VGG16** delivered the best validation accuracy and generalization, closely followed by **ResNet50**.

---

## 📁 Dataset Structure

The dataset was organized in the following structure:

dataset/ ├── train/ │ ├── fresh/ │ └── rotten/ ├── test/ │ ├── fresh/ │ └── rotten/

yaml
Copy
Edit

Images were augmented using `ImageDataGenerator` for better generalization.

---

## ⚙️ Technologies Used

- Python 🐍
- TensorFlow / Keras
- NumPy, Matplotlib
- CNN, VGG16, ResNet50
- Jupyter Notebook

---

---

## 📊 Model Performance

### ✅ Accuracy & Loss

After training the model for 10 epochs:

- **Training Accuracy:** ~99.59%
- **Validation Accuracy:** ~96.25%
- **Training Loss:** ~0.012
- **Validation Loss:** ~0.125

These results indicate that the model generalizes well and can effectively classify unseen data with high precision.
![Sample Prediction](https://github.com/Zem-0/Classification_of_fruits/blob/main/Screenshot%202025-04-06%20195902.png)
<p align="center">
  <img src="https://github.com/Zem-0/Classification_of_fruits/blob/main/download%20(4).png" width="30%" />
  <img src="https://github.com/Zem-0/Classification_of_fruits/blob/main/download%20(5).png" width="30%" />
  <img src="https://github.com/Zem-0/Classification_of_fruits/blob/main/download%20(6).png" width="30%" />
</p>




## ⚙️ Technologies Used

- Python 🐍
- TensorFlow / Keras
- Matplotlib
- NumPy
- ImageDataGenerator (for data augmentation)

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fresh-rotten-fruit-classifier.git
   cd fresh-rotten-fruit-classifier
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook:

bash
Copy
Edit
jupyter notebook Fresh_rotten_Classifier.ipynb
🖼️ Sample Predictions
The model was tested with new fruit images and accurately classified them as either fresh or rotten. Below are a few predictions made by the model:

Input Image	Predicted Class
🍌 Rotten Banana	Rotten
🍎 Shiny Apple	Fresh
🍇 Moldy Grapes	Rotten
🏁 Future Work
Deploy as a mobile or web app using TensorFlow Lite or Flask

Train on more classes (e.g., specific fruit types like bananas, apples, oranges)

Use object detection to identify multiple fruits in a single image

🙌 Author
Parinith
📧 parinith99@gmail.com
🌐 GitHub
