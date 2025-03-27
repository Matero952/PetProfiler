import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.openimages

# Download images for specific classes

classes = fo.utils.openimages.get_classes()
print(f"Classes available: {classes}")
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    label_types=["classifications"],  # You can also use "classifications" for single labels
    classes=["Dog", "Man"],
    max_samples=200,  # Adjust as needed
)

# Visualize dataset
classes = fo.utils.openimages.get_classes()
print(f"Classes available: {classes}")
session = fo.launch_app(dataset)

