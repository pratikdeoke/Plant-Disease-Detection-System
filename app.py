from flask import Flask, request, render_template, url_for
import os
import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

# Load your trained model
model = load_model('D:\Major Project\Project 4\my_model4.h5')

class_mapping = {
    0: "Apple scab",
    1: "Apple Black rot",
    2: "Cedar Apple rust",
    3: "Apple healthy",
    4: "Blueberry healthy",
    5: "Cherry Powdery mildew",
    6: "Cherry healthy",
    7: "Corn (maize) Cercospora leaf spot Gray leaf spot",
    8: "Corn (maize) Common rust",
    9: "Corn (maize) Northern Leaf Blight",
    10: "Corn (maize) healthy",
    11: "Grape Black rot",
    12: "Grape Esca (Black Measles)",
    13: "Grape Leaf blight (Isariopsis Leaf Spot)",
    14: "Grape healthy",
    15: "Orange Haunglongbing (Citrus greening)",
    16: "Peach Bacterial spot",
    17: "Peach healthy",
    18: "Pepper, bell Bacterial spot",
    19: "Pepper, bell healthy",
    20: "Potato Early blight",
    21: "Potato Late blight",
    22: "Potato healthy",
    23: "Raspberry healthy",
    24: "Soybean healthy",
    25: "Squash Powdery mildew",
    26: "Strawberry Leaf scorch",
    27: "Strawberry healthy",
    28: "Tomato Bacterial spot",
    29: "Tomato Early blight",
    30: "Tomato Late blight",
    31: "Tomato Leaf Mold",
    32: "Tomato Septoria leaf spot",
    33: "Tomato Spider mites Two spotted spider mite",
    34: "Tomato Target Spot",
    35: "Tomato Tomato Yellow Leaf Curl Virus",
    36: "Tomato Tomato mosaic virus",
    37: "Tomato healthy"
}

@app.route('/', methods=['GET'])
def home():
    if request.method == 'POST':
        if 'file' in request.files:
            # Single image prediction
            return predict_single()
        elif 'folder' in request.files:
            # Folder prediction
            return predict_folder()
    # Render the home page with the upload form
    return render_template('homepage.html')

@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/aboutus')
def about_us():
    return render_template('aboutus.html')

@app.route('/contactus')
def _us():
    return render_template('contactus.html')

@app.route('/single_image_prediction', methods=['POST'])
def single_image_prediction():
    if 'file' in request.files:
        # Single image prediction
        return predict_single()
    else:
        return render_template('homepage.html', prediction='No file selected for single image prediction')

@app.route('/folder_prediction', methods=['POST'])
def folder_prediction():
    if 'folder' in request.files:
        # Folder prediction
        return predict_folder()
    else:
        return render_template('homepage.html', prediction='No folder selected for folder prediction')

def predict_single():
    if 'file' not in request.files:
        return render_template('homepage.html', prediction='There is no file in form!')
    file = request.files['file']
    if file.filename == '':
        return render_template('homepage.html', prediction='No selected file')
    if file:
        # Save the file to the server
        filepath = './uploads/' + file.filename
        file.save(filepath)
        
        # Load the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array_expanded_dims)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_mapping.get(predicted_class_index, "Unknown class")

        # Generate URL for the uploaded image
        image_url = url_for('static', filename='uploads/' + file.filename)

        # Return the predicted class and image URL to the template
        return render_template('homepage.html', prediction=predicted_class_name, image_url=image_url)

def predict_folder():
    if 'folder' not in request.files:
        return render_template('homepage.html', prediction='There is no folder in form!')
    folder = request.files.getlist('folder')
    if not folder:
        return render_template('homepage.html', prediction='No selected folder')
    
    predictions = []
    for file in folder:
        # Get the folder name from the file path
        folder_name = os.path.basename(os.path.dirname(file.filename))
        
        # Create the folder if it doesn't exist
        folder_path = os.path.join('uploads', folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Save the file to the server
        filepath = os.path.join(folder_path, os.path.basename(file.filename))
        file.save(filepath)
        
        # Load the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array_expanded_dims)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_mapping.get(predicted_class_index, "Unknown class")
        
        print("Predicted class name:", predicted_class_name)  # Debug print
        
        # Generate URL for the uploaded image
        image_url = url_for('static', filename=os.path.join('uploads', folder_name, os.path.basename(file.filename)))
        
        # Append prediction and image URL to the list
        predictions.append((predicted_class_name, image_url))

    print("Predictions:", predictions)  # Debug print

    # Return the predicted classes and image URLs to the template
    return render_template('homepage.html', predictions=predictions)


if __name__ == '__main__':
    app.run(debug=True)
