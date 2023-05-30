import json

def replace_image_extension(image_labels:list):
    for frame in image_labels:
        if frame["filename"].endswith(".tiff"):
            frame["filename"] = frame["filename"].replace(".tiff", ".png")
    return image_labels

def import_from_json(filename):
    with open(filename) as f:
        # loads data as list of dicts (list of frames)
        frames = json.load(f)
        return frames
