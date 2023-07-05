import json
import os
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


def export_to_json(folder,filename,file):
    json_export = json.dumps(file)
    path = os.path.join(folder, filename)
    with open(f'{path}.json', 'w') as outfile:
        outfile.write(json_export)