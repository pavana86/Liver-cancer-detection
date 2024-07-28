from django.shortcuts import render

# Create your views here.
import os
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.conf import settings
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the model
model = load_model('C:/Users/pavan/Downloads/liver_cancer_detection_cnn_model.h5')  # Ensure the path to your model is correct

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def index(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        default_storage.save(file_path, file)

        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        class_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3']
        result = class_names[np.argmax(prediction)]

        return render(request, 'classifier/result.html', {
            'result': result,
            'image_url': settings.MEDIA_URL + file.name
        })

    return render(request, 'classifier/index.html')
