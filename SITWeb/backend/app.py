# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ask_chatbot

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    print("[DEBUG] Incoming request:", data)
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    try:
        response_text = ask_chatbot(prompt)
        print("[DEBUG] Generated response:", response_text)
        return jsonify({"response": response_text})
    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)