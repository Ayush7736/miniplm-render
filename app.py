from flask import Flask, request, jsonify
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

model_path = hf_hub_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
)

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_threads=2
)

@app.route("/")
def home():
    return "AI running"

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    print("USER:", data, flush=True)

    user_msg = data.get("message","")

    output = llm(
        f"You are a helpful assistant.\nUser: {user_msg}\nAssistant:",
        max_tokens=120,
        temperature=0.7
    )

    reply = output["choices"][0]["text"]

    print("AI:", reply, flush=True)

    return jsonify({"reply": reply})

port = int(os.environ.get("PORT", 10000))

app.run(host="0.0.0.0", port=port)
