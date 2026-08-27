import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from PIL import Image

    import torch as torch

    return AutoImageProcessor, AutoModelForImageClassification, Image, torch


@app.cell
def _(AutoImageProcessor, AutoModelForImageClassification):
    # Variables
    file_img = 'Welwitschia mirabilis.jpg'

    # Paths
    path_img = 'img/'

    # Load the model and processor from Hugging Face
    model_id = "phenobase/phenovision"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    return file_img, model, path_img, processor


@app.cell
def _(Image, file_img, path_img):
    # Open a local image of a plant specimen
    image = Image.open(path_img + file_img)
    return (image,)


@app.cell
def _(image, model, processor, torch):
    # Prepare inputs and run inference
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0]
    # IMPORTANT: output order is [fruit, flower] — index 0 is FRUIT, index 1 is FLOWER.
    fruit_prob  = probs[0].item()
    flower_prob = probs[1].item()
    print(f"Flower: {flower_prob:.3f}")
    print(f"Fruit:  {fruit_prob:.3f}")
    return


if __name__ == "__main__":
    app.run()
