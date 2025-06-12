# backend/app.py
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from chatbot_caching import ask_chatbot

app = Flask(__name__)
CORS(app)

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

if __name__ == "__main__":
    app.run(debug=True, port=5000)