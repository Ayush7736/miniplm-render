from flask import Flask, request, jsonify
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# download GGUF model automatically
model_path = hf_hub_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
)

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_threads=2
)

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]

    output = llm(
        f"User: {user_msg}\nAssistant:",
        max_tokens=100,
        stop=["User:"]
    )

    reply = output["choices"][0]["text"]

    return jsonify({"reply": reply})

app.run(host="0.0.0.0", port=10000)
