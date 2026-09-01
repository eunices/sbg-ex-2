import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from PIL import Image
    from IPython.display import display

    import torch as torch

    return (
        AutoImageProcessor,
        AutoModelForImageClassification,
        Image,
        display,
        torch,
    )


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
def _(Image, display, model, path_img, processor, torch):
    # Open a local image of a plant specimen
    # true positive: 'BM000521866.jpg', 'bad.jpg'
    # true negative: '270005.jpg' 
    # false positive: '196181.jpg'
    # false negative: '269465.jpg'

    file_img = 'bad.jpg' 
    image = Image.open(path_img + file_img)
    display(image, height= 300)

    # Prepare inputs and run inference
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0]

    fruit_prob  = probs[0].item()
    flower_prob = probs[1].item()
    print(f"Flower: {flower_prob:.3f}")
    # print(f"Fruit:  {fruit_prob:.3f}")

    # test on weird plants with flowers:
    # can detect flower:  'Welwitschia mirabilis.jpg', 'Pleurothallis-truncata.jpg'
    # cannot detect flower: 'Picture2.jpg'
    return


if __name__ == "__main__":
    app.run()
