from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.hazmat.primitives import serialization
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models/signature_model.h5'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model(MODEL_PATH)

# RSA Key Generation
def generate_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    # Serialize the keys
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem

# Signature Preprocessing
def preprocess_signature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=-1)  # Add channel
    img = np.expand_dims(img, axis=0)   # Add batch
    img = img / 255.0
    return img

# Document Signing
def sign_document(document: bytes, private_key_pem: bytes):
    private_key = load_pem_private_key(private_key_pem, password=None)
    signature = private_key.sign(
        document,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    return signature

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Save uploaded files
        signature_file = request.files['signature']
        doc_file = request.files['document']
        signature_path = os.path.join(UPLOAD_FOLDER, signature_file.filename)
        doc_path = os.path.join(UPLOAD_FOLDER, doc_file.filename)

        signature_file.save(signature_path)
        doc_file.save(doc_path)

        # Preprocess and verify
        img = preprocess_signature(signature_path)
        prediction = model.predict(img)

        # Generate RSA keys
        private_key, public_key = generate_keys()

        # Perform signing if signature is verified
        if prediction > 0.5:
            with open(doc_path, 'rb') as f:
                document_data = f.read()
            signed_doc = sign_document(document_data, private_key)
            return render_template('result.html', result='Signature Verified and Document Signed!')
        else:
            return render_template('result.html', result='Signature Verification Failed!')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
