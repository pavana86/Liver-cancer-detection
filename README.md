🧬 Liver Cancer Detection using Deep Learning

This project presents a deep learning-based web application for automated **liver cancer detection** using **histopathology images**. The goal is to assist medical professionals by providing accurate predictions, helping with early diagnosis and timely treatment.

 🚀 Features

- 📊 **Automated Cancer Detection** using CNN & EfficientNetB3 models  
- 🔍 **Preprocessed Histopathology Images** using image enhancement techniques  
- 🌐 **User-friendly Django Web App** for image upload and result visualization  
- 📈 Achieved **>85% accuracy** on validation dataset  
- 📁 Secure and local image handling (no third-party API use)  

🧠 Model Architecture

- **CNN** and **EfficientNetB3** used for high-performance image classification  
- Training with **categorical cross-entropy**, **Adam optimizer**, and **early stopping**  
- Input images are normalized, resized, and augmented for improved generalization  


🛠 Tech Stack

| Component           | Tech Used                 |
|---------------------|---------------------------|
| Language            | Python 3.11               |
| Framework           | Django                    |
| Deep Learning       | TensorFlow / Keras        |
| Image Processing    | OpenCV, PIL               |
| Front-end (UI)      | HTML, CSS, Bootstrap      |


🖼 Sample UI

- Upload histopathology image  
- Get instant prediction result (Cancerous / Non-Cancerous)  
- View prediction confidence score

 📁 Folder Structure

```
lcd-project/
├── media/                 # Uploaded images
├── static/                # CSS and JS
├── templates/             # HTML Templates
├── model/                 # Saved model files
├── lcd_app/               # Django app
│   └── views.py, models.py, urls.py
├── manage.py
└── README.md
```
📌 Future Improvements

- Add more data from diverse sources  
- Implement Grad-CAM for explainability  
- Deploy on cloud (e.g., AWS or Heroku)

🙋‍♀️ Author

**Pavana M**  
Email: [mpavana8603@gmail.com]  
LinkedIn: [www.linkedin.com/in/pavana-m-668663292]  
GitHub: [https://github.com/pavana86]
