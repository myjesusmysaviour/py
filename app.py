from flask import Flask, request, jsonify, send_file
from gradio_client import Client
from flask_cors import CORS
import base64
import os
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the client
try:
    client = Client("OpenSound/EzAudio")
    logger.info("Gradio client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gradio client: {str(e)}", exc_info=True)
    client = None

@app.route('/', methods=['GET'])
def index():
    return send_file('index.html')

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    if client is None:
        logger.error("Gradio client is not initialized")
        return jsonify({"error": "Service is not available"}), 503

    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        logger.info(f"Attempting to generate audio for text: {text}")

        # Generate audio
        result = client.predict(
            text,  # text
            8,  # length
            5,  # guidance_scale
            0.75,  # guidance_rescale
            50,  # ddim_steps
            1,  # eta
            0,  # random_seed
            True,  # randomize_seed
            api_name="/generate_audio"
        )

        logger.info("Audio generated successfully")

        # Encode the audio file to base64
        with open(result, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

        # Remove the temporary file
        os.remove(result)

        return jsonify({"audio": encoded_audio})
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while generating the audio. Please check server logs for details."}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

# Remove the if __name__ == '__main__' block for Vercel deployment