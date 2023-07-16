import random
import torchvision.transforms as T

def augment_image(image_cell,probability:float):
    augmented_tensor =image_cell

    if random.random() < probability:
        augmented_tensor = T.GaussianBlur(kernel_size=3, sigma=0.75)(
            augmented_tensor)

    if random.random() < probability:
        augmented_tensor = T.RandomAdjustSharpness(sharpness_factor=2)(
            augmented_tensor)

    if random.random() < probability:
        augmented_tensor = T.RandomHorizontalFlip()(augmented_tensor)

    if random.random() < probability:
        augmented_tensor = T.RandomVerticalFlip()(augmented_tensor)

    # if random.random() < probability:
    #     augmented_tensor = T.RandomGrayscale()(augmented_tensor)


    return augmented_tensor
