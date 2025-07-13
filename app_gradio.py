import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime as rt
from PIL import Image
import requests
from tqdm import tqdm
import gradio as gr

MODELS = {
    "wd-eva02-large-tagger-v3": {
        "onnx": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx",
        "safetensors": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.safetensors",
        "tags": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv",
    },
    "wd-swinv2-tagger-v3": {
        "onnx": "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx",
        "safetensors": "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.safetensors",
        "tags": "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv",
    },
}

MODELS_DIR = "./Models"
ALL_PROMPTS = "all_prompts.txt"


def download_with_progress(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        return
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {os.path.basename(dest)}") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def ensure_resources(model_key: str):
    info = MODELS[model_key]
    onnx_path = os.path.join(MODELS_DIR, model_key + ".onnx")
    safe_path = os.path.join(MODELS_DIR, model_key + ".safetensors")
    tags_path = os.path.join(MODELS_DIR, "selected_tags.csv")
    download_with_progress(info["onnx"], onnx_path)
    download_with_progress(info["safetensors"], safe_path)
    download_with_progress(info["tags"], tags_path)
    return onnx_path, tags_path


def load_model(path: str):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = rt.InferenceSession(path, providers=providers)
    return session


def load_labels(csv_path: str):
    df = pd.read_csv(csv_path)
    tag_names = df["name"].tolist()
    rating_indexes = list(np.where(df["category"] == 9)[0])
    general_indexes = list(np.where(df["category"] == 0)[0])
    character_indexes = list(np.where(df["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def predict(session, image: Image.Image, tag_names, rating_indexes, general_indexes, character_indexes, general_threshold=0.5, character_threshold=0.85):
    _, height, width, _ = session.get_inputs()[0].shape
    image = image.convert("RGBA")
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    image = np.asarray(image)
    image = image[:, :, ::-1]
    image_size = (height, height)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    probs = session.run([label_name], {input_name: image})[0]

    labels = list(zip(tag_names, probs[0].astype(float)))
    ratings_names = [labels[i] for i in rating_indexes]
    rating = dict(ratings_names)
    general_names = [labels[i] for i in general_indexes]
    general_res = [x for x in general_names if x[1] > general_threshold]
    general_res = dict(general_res)
    character_names = [labels[i] for i in character_indexes]
    character_res = [x for x in character_names if x[1] > character_threshold]
    character_res = dict(character_res)
    tags_sorted = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
    result = ", ".join(list(tags_sorted.keys())).replace("_", " ")
    return result


def process_images(files, model_name):
    if not files:
        return "No files provided"
    onnx_path, tags_path = ensure_resources(model_name)
    session = load_model(onnx_path)
    tag_names, rating_indexes, general_indexes, character_indexes = load_labels(tags_path)
    results = []
    for file in files:
        image = Image.open(file.name)
        res = predict(session, image, tag_names, rating_indexes, general_indexes, character_indexes)
        txt_path = os.path.splitext(file.name)[0] + ".txt"
        with open(txt_path, "w") as f:
            f.write(res)
        results.append(f"{os.path.basename(file.name)}: {res}")
    with open(ALL_PROMPTS, "a") as f:
        for r in results:
            f.write(r + "\n\n")
    return "\n".join(results)


def main():
    with gr.Blocks() as demo:
        gr.Markdown("## WD14 Batch Tagger")
        model_dropdown = gr.Dropdown(list(MODELS.keys()), value="wd-swinv2-tagger-v3", label="Model")
        image_input = gr.File(file_types=[".png", ".jpg", ".jpeg"], label="Images", multiple=True)
        output = gr.Textbox(label="Tags")
        run_btn = gr.Button("Tag Images")
        run_btn.click(fn=process_images, inputs=[image_input, model_dropdown], outputs=output)
    demo.launch()


if __name__ == "__main__":
    main()
