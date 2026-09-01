import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from PIL import Image
    from IPython.display import display

    import torch as torch

    return AutoImageProcessor, AutoModelForImageClassification, Image, torch


@app.cell
def _():
    # Paths
    path_img = 'img/'
    return (path_img,)


@app.cell
def _(AutoImageProcessor, AutoModelForImageClassification):
    # Load the model and processor from Hugging Face
    model_id = "phenobase/phenovision"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    return model, processor


@app.cell
def _(Image, model, path_img, processor, torch):
    # Open a local image of a plant specimen
    # true positive: '1.jpg', '2.jpg'
    # true negative: '3.jpg' 
    # false positive: '4.jpg', 'poaceae.jpg'
    # false negative: '5.jpg'

    file_img = '1.jpg' 
    image = Image.open(path_img + file_img)

    # Prepare inputs and run inference
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0]

    fruit_prob  = probs[0].item()
    flower_prob = probs[1].item()
    print(f"Flower: {flower_prob:.3f}")
    # print(f"Fruit:  {fruit_prob:.3f}")

    # display(image)
    return


if __name__ == "__main__":
    app.run()
