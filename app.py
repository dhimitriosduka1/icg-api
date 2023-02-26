import logging

import yaml
from flask import Flask, request, jsonify

from service.model_service import init_model, init_transform, predict

app = Flask(__name__)

logging.getLogger().setLevel(logging.INFO)

logging.info("Loading config file")
with open("config/model_config.yml", "r") as f:
    config = yaml.safe_load(f)
logging.info("Successfully loaded config file")

model, vocabulary = init_model(config)
transform = init_transform(config)


@app.route("/predict", methods=["POST"])
def get_prediction():
    image = request.files["image"]
    caption = predict(model, transform, image, vocabulary)
    return jsonify({"caption": str(caption)})


if __name__ == "__main__":
    app.run()
