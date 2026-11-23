import gradio as gr
import torch
import mimetypes
from PIL import Image
from torchvision import transforms
from convnext_model import load_convnext_checkpoint
from image_preprocessing import crop_largest_face

UNCERTAINTY_THRESHOLD = 0.80

# === Load Model ===
def load_model():
    return load_convnext_checkpoint("models/convnext_tiny_best.ckpt")

model = load_model()

# === Preprocessing ===
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Inference Logic ===
def predict_file(file_obj):
    if file_obj is None:
        return "⚠️ No file selected", "", None

    path = file_obj.name
    mime, _ = mimetypes.guess_type(path)

    if mime and mime.startswith("image"):
        img = Image.open(path).convert("RGB")
        img = crop_largest_face(img)
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)
        confidence = conf.item()
        if confidence < UNCERTAINTY_THRESHOLD:
            label = "🟡 Inconclusive"
        else:
            label = "🟢 Real" if pred.item() == 0 else "🔴 Deepfake"
        return (
            f"{label}",
            f"{confidence * 100:.2f}%",
            img,
        )

    else:
        return "Unsupported file type", "", None

# === Gradio UI ===
with gr.Blocks(title="Deepfake Image Detector") as demo:
    gr.Markdown("## 🧠 Deepfake Image Detector\nDrop in an image.")

    file_input = gr.File(
        label="Drop The Image Here",
        file_types=[".jpg", ".jpeg", ".png"],
    )
    

    with gr.Row():
        prediction = gr.Textbox(label="Prediction", interactive=False)

    preview = gr.Image(label="Preview", interactive=False)

    def handle_input(file_obj):
        return predict_file(file_obj)

    file_input.change(
        fn=handle_input,
        inputs=file_input,
        outputs=[prediction, preview]
    )

demo.launch()
