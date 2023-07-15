import random
from torchvision import transforms


def augment_image(image_tensor):
    # Convert image tensor to PIL Image
    image = transforms.ToPILImage()(image_tensor)

    # Randomly select an augmentation technique
    augmentation = random.choice([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        # Add more augmentation techniques as desired
    ])

    # Apply the selected augmentation technique to the image
    augmented_image = augmentation(image)
    resized_image = augmented_image.resize((64, 64))
    # Convert the resized PIL Image back to tensor
    resized_tensor = transforms.ToTensor()(resized_image)
    return resized_tensor
