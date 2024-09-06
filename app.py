from flask import Flask, request, render_template, send_from_directory, jsonify, url_for
import os
import torchvision
import torch
import numpy as np
import random
import string
import cv2
from PIL import Image
from torchvision import transforms


# Function to generate a unique filename
def generate_unique_filename(extension="png"):
    characters = string.ascii_letters + string.digits
    unique_name = ''.join(random.choice(characters) for _ in range(8))
    return f"{unique_name}.{extension}"


# Load the pre-trained DeepLabV3 ResNet model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Preprocess the input image
def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/process-image', methods=['POST'])
def process_image():
    # Validate the app key
    app_key = request.form.get('app_key')
    if app_key != 'mlovek143rdn2005':
        return jsonify({'status': 'error', 'message': 'Invalid app key'}), 401
    
    # Validate the file
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file provided'}), 400
    
    file = request.files['image']
    if file:
        # Generate a unique filename for the uploaded image
        unique_image_name = generate_unique_filename(extension=file.filename.split('.')[-1])
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_image_name)
        file.save(image_path)
                    
        # Load and preprocess the image
        image = Image.open(image_path)
        original_width, original_height = image.size
        input_tensor = preprocess(image)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
            output_predictions = output.argmax(0)

            # Convert output_predictions to a NumPy array and resize to original image dimensions
            mask = output_predictions.byte().cpu().numpy()
            mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

            # Convert the mask to a binary format (0 or 255)
            mask = (mask * 255).astype(np.uint8)

            # Convert the original image to a NumPy array
            image_np = np.array(image)

            # Ensure the image is in the BGR format (for OpenCV)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Apply the mask using bitwise_and
            final_image = cv2.bitwise_and(image_np, image_np, mask=mask)

            # Save the final image
            # Generate a unique filename for the result image
            result_image_name = generate_unique_filename(extension="png")
            save_path = os.path.join(app.config['RESULT_FOLDER'], result_image_name)
            cv2.imwrite(save_path, final_image)

            # Generate the URL for the saved result image
            result_image_url = url_for('static', filename=f'results/{result_image_name}', _external=True)

            # Return JSON response with the image URL and status message
            return jsonify({'status': 'success', 'image_url': result_image_url}), 200          

    return jsonify({'status': 'error', 'message': 'Image processing failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
