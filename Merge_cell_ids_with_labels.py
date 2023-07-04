import os
import json
from helper_functions import import_from_json

full_images_with_ages = import_from_json("full_dataset_with_ages.json")
full_images_with_ids = import_from_json("full_dataset_predictions_updated.json")
# folder_path = "./training_tensor_data/labels"  # Replace with the actual folder path
#
# # List all files in the folder
# files = os.listdir(folder_path)
def add_ids_tojson_with_ages():
    for image_with_id in full_images_with_ids:
        image_name_with_id = os.path.splitext(image_with_id.get("filename"))[0]
        cells_with_ids = image_with_id.get("cells")
        for image_with_age in full_images_with_ages:
            image_name_with_age = os.path.splitext(image_with_age.get("filename"))[0]
            if image_name_with_id == image_name_with_age:
                for i, row_cells in enumerate(image_with_age.get("cells")):
                    for j,cell in enumerate(row_cells):
                        cell_details = cells_with_ids[i][j]
                        cell["cell_id"] = cell_details.get("cell_id")
    return full_images_with_ages


def export_to_json(full_dataset_with_ids):
    json_export = json.dumps(full_dataset_with_ids)
    with open('full_dataset_with_ages_with_ids.json', 'w') as outfile:
        outfile.write(json_export)

def main():
    json_with_id_and_age = add_ids_tojson_with_ages()
    export_to_json(json_with_id_and_age)

if __name__ == "__main__":
    main()

