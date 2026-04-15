from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForCausalLM.from_pretrained("./model")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json["message"]

    inputs = tokenizer(user_msg, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=60)

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"reply": response})

app.run(host="0.0.0.0", port=10000)
