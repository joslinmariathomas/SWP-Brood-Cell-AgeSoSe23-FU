import random
from torchvision import transforms

def augment_image(image_tensor):
    # Convert image tensor to PIL Image
    image = transforms.ToPILImage()(image_tensor)

    # Randomly select an augmentation technique
    augmentation = random.choice([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        # Add more augmentation techniques as desired
    ])

    # Apply the selected augmentation technique to the image
    augmented_image = augmentation(image)

    # Convert the augmented PIL Image back to tensor
    augmented_tensor = transforms.ToTensor()(augmented_image)
    return augmented_tensor
