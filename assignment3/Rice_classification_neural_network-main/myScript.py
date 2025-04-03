import sys
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the pretrained model
model = torch.load('model.pth')

# Set the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Preprocess the input image
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor()
])

def predict_rice_type(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Make the prediction
    with torch.no_grad():
        model.eval()
        output = model(image)

    # Get the predicted rice type
    _, predicted = torch.max(output, 1)
    predicted_label = predicted.item()

    # Map the predicted label to the rice type
    labels_map = {
        0: "Arborio",
        1: "Basmati",
        2: "Ipsala",
        3: "Jasmine",
        4: "Karacadag"
    }

    predicted_type = labels_map[predicted_label]

    return predicted_type

# Check if the image path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the path to the image as a command-line argument.")
    sys.exit(1)

# Get the image file path from the command-line argument
image_file = sys.argv[1]

# Predict the rice type
predicted_type = predict_rice_type(image_file)
print(predicted_type)
