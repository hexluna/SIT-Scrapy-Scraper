# backend/app.py
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from chatbot_caching import ask_chatbot
from embedding_pipeline import IncrementalEmbedder
import os
import threading
import tempfile

app = Flask(__name__)
CORS(app)
embedder = IncrementalEmbedder()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    print("[DEBUG] Incoming request:", data)
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    def generate():
        try:
            for chunk in ask_chatbot(prompt):
                yield chunk
        except Exception as e:
            yield f"[ERROR]: {str(e)}"

    return Response(stream_with_context(generate()), content_type='text/plain')

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if not file or not file.filename.endswith(".pdf"):
        return "Only PDF files are supported.", 400

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)

    def generate_stream():
        try:
            for token in embedder._generate_summary(temp_path):
                yield token
        except Exception as e:
            yield f"[ERROR]: {str(e)}"

    return Response(stream_with_context(generate_stream()), content_type='text/plain')

if __name__ == "__main__":
    app.run(debug=True, port=5000)