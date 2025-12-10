# Confusion Matrix and ROC Curve Analysis
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from torchvision import transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.font_manager as fm

# Create directory for saving results
result_dir = 'E:\WSI-HSfeature/analysis_results'
os.makedirs(result_dir, exist_ok=True)

# Set font to Arial Bold
font_path = None
# Try to find Arial font path
for font in fm.fontManager.ttflist:
    if 'arial' in font.name.lower():
        font_path = font.fname
        break

if font_path:
    arial_font = fm.FontProperties(fname=font_path, weight='bold')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
else:
    print("Arial font not found, using default font")

plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


# Define Dataset Class
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        imagelist = []
        labelist = []
        # Check if img_dir exists
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Directory does not exist: {img_dir}")

        # Traverse all subdirectories
        for folder in os.listdir(img_dir):
            folder_path = os.path.join(img_dir, folder)
            # Only process directories
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    # Ensure processing files not subdirectories
                    if os.path.isfile(file_path):
                        imagelist.append(file)
                        labelist.append(folder)

        self.imagelist = imagelist
        self.labelist = labelist
        print(f"Loaded {len(imagelist)} images, divided into {len(set(labelist))} classes")

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labelist[idx], self.imagelist[idx])
        # Check if file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file does not exist: {img_path}")

        image = Image.open(img_path)
        labels_map = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}
        label = labels_map[self.labelist[idx]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Test Dataset
try:
    test_dataset = ImageDataset(r'E:\WSI-HSfeature\CRC-VAL-HE-7K', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Test dataset loaded successfully, total {len(test_dataset)} samples")
except Exception as e:
    print(f"Error loading test dataset: {e}")
    # Add fallback or exit if loading fails
    import sys
    sys.exit(1)

# Load Model
model = models.resnet50(weights=None)
# Modify the last layer to match our number of classes
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 1024),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(1024, 9)
)

try:
    model.load_state_dict(torch.load('E:\WSI-HSfeature/resnet50_weights.pth'))
    print("Model weights loaded successfully")
except Exception as e:
    print(f"Error loading model weights: {e}")
    sys.exit(1)

model = model.to(device)
model.eval()

# Get Model Predictions
y_true = []
y_score = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        # Use softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_score.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_score = np.array(y_score)

# Binarize labels
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
n_classes = y_true_bin.shape[1]

# Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Class label mapping
labels_map = {0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 4: "MUC", 5: "MUS", 6: "NORM", 7: "STR", 8: "TUM"}

# Plot ROC curve for each class (separate plots)
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
for i, color in zip(range(n_classes), colors):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr[i], tpr[i], color=color, lw=3,
             label=f'ROC curve (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve for {labels_map[i]} class', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 10})
    plt.grid(True)

    # Save image
    plt.savefig(os.path.join(result_dir, f'ROC_{labels_map[i]}.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Calculate Confusion Matrix
y_pred = np.argmax(y_score, axis=1)
cm = confusion_matrix(y_true, y_pred)

# Print Classification Report
class_names = list(labels_map.values())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Visualize Confusion Matrix (adjust font size and style)
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, fontsize=12, fontweight='bold')
plt.yticks(tick_marks, class_names, fontsize=12, fontweight='bold')

# Annotate values on confusion matrix (adjust font size)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 fontsize=12, fontweight='bold',
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True Label', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"All results saved to {result_dir} folder")

# Classification ROC Curves
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from torchvision import transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.font_manager as fm

# Create directory for saving results
result_dir = 'E:\WSI-HSfeature/analysis_results'
os.makedirs(result_dir, exist_ok=True)


# Font Configuration Class
class FontConfig:
    def __init__(self, font_size=35, font_weight='bold', font_family=None):
        self.font_size = font_size
        self.font_weight = font_weight
        self.font_family = font_family
        self.font_path = None

    def setup_font(self):
        """Configure matplotlib font"""
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['font.weight'] = self.font_weight
        plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

        # Font lookup logic
        if self.font_family:
            plt.rcParams['font.family'] = self.font_family
            return True

        try:
            # Prioritize finding Arial font
            for font in fm.fontManager.ttflist:
                if 'arial' in font.name.lower():
                    self.font_path = font.fname
                    break
            # If Arial not found, try other common sans-serif fonts
            if not self.font_path:
                for font in fm.fontManager.ttflist:
                    if 'sans-serif' in font.style.lower() or 'helvetica' in font.name.lower():
                        self.font_path = font.fname
                        break
        except Exception as e:
            print(f"Font lookup error: {e}")
            return False

        if self.font_path:
            plt.rcParams['font.family'] = 'Arial'
            print(f"Font set to: Arial, Size: {self.font_size}, Weight: {self.font_weight}")
            return True
        else:
            print(f"Specified font not found, using default font, Size: {self.font_size}, Weight: {self.font_weight}")
            return False


# Define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


# Define Dataset Class
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        imagelist = []
        labelist = []
        # Check if img_dir exists
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Directory does not exist: {img_dir}")

        # Traverse all subdirectories
        for folder in os.listdir(img_dir):
            folder_path = os.path.join(img_dir, folder)
            # Only process directories
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    # Ensure processing files not subdirectories
                    if os.path.isfile(file_path):
                        imagelist.append(file)
                        labelist.append(folder)

        self.imagelist = imagelist
        self.labelist = labelist
        print(f"Loaded {len(imagelist)} images, divided into {len(set(labelist))} classes")

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labelist[idx], self.imagelist[idx])
        # Check if file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file does not exist: {img_path}")

        image = Image.open(img_path)
        labels_map = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}
        label = labels_map[self.labelist[idx]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Evaluation Class
class ModelEvaluator:
    def __init__(self, model, test_dataloader, result_dir, font_config=None):
        self.model = model
        self.test_dataloader = test_dataloader
        self.result_dir = result_dir
        self.font_config = font_config or FontConfig()
        self.labels_map = {0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 4: "MUC", 5: "MUS", 6: "NORM", 7: "STR", 8: "TUM"}
        self.y_true = None
        self.y_score = None

    def evaluate(self, save_predictions=True):
        """Perform model evaluation and save results"""
        # Set font
        self.font_config.setup_font()

        # Get model predictions
        self.y_true = []
        self.y_score = []

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                # Use softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                self.y_true.extend(labels.cpu().numpy())
                self.y_score.extend(probs.cpu().numpy())

        self.y_true = np.array(self.y_true)
        self.y_score = np.array(self.y_score)

        # Save prediction results
        if save_predictions:
            np.save(os.path.join(self.result_dir, 'y_true.npy'), self.y_true)
            np.save(os.path.join(self.result_dir, 'y_score.npy'), self.y_score)
            print(f"Prediction results saved to {self.result_dir}")

        return self.y_true, self.y_score

    def plot_roc_curves(self):
        """Plot ROC curves (no title, only axis labels, enlarged AUC label)"""
        if self.y_true is None or self.y_score is None:
            print("Please perform evaluation first")
            return

        # Binarize labels
        y_true_bin = label_binarize(self.y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        n_classes = y_true_bin.shape[1]

        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
        for i, color in zip(range(n_classes), colors):
            plt.figure(figsize=(12, 10))
            plt.plot(fpr[i], tpr[i], color=color, lw=3,
                     label=f'ROC curve (AUC = {roc_auc[i]:.5f})')  # Original AUC label
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])

            # Enlarged AUC label (added next to curve)
            plt.annotate(f'AUC = {roc_auc[i]:.5f}',
                         xy=(0.3, 0.6),  # Label position, adjust as needed
                         xytext=(0.3, 0.6),
                         fontsize=self.font_config.font_size + 10,  # Enlarge font by 10 points
                         fontweight=self.font_config.font_weight,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            # Set axis labels (enlarged font)
            plt.xlabel('False Positive Rate (FPR)', fontsize=self.font_config.font_size + 4,
                       fontweight=self.font_config.font_weight)
            plt.ylabel('True Positive Rate (TPR)', fontsize=self.font_config.font_size + 4,
                       fontweight=self.font_config.font_weight)

            # Remove title
            # plt.title(f'ROC Curve for {self.labels_map[i]} class', fontsize=self.font_config.font_size+4, fontweight=self.font_config.font_weight)

            # Keep legend but hide AUC (already annotated on curve)
            plt.legend(loc="lower right",
                       prop={'weight': self.font_config.font_weight, 'size': self.font_config.font_size})

            plt.grid(True, linestyle='--', alpha=0.7)

            # Add coordinate grid lines
            plt.minorticks_on()
            plt.grid(which='major', axis='both', linestyle='-', linewidth=0.5)
            plt.grid(which='minor', axis='both', linestyle=':', linewidth=0.3)

            # Save image
            plt.savefig(os.path.join(self.result_dir, f'ROC_{self.labels_map[i]}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Generated ROC curves for {len(self.labels_map)} classes")

    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        if self.y_true is None or self.y_score is None:
            print("Please perform evaluation first")
            return

        # Calculate confusion matrix
        y_pred = np.argmax(self.y_score, axis=1)
        cm = confusion_matrix(self.y_true, y_pred)
        class_names = list(self.labels_map.values())

        # Visualize confusion matrix
        plt.figure(figsize=(16, 14))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=self.font_config.font_size + 6, fontweight=self.font_config.font_weight)
        plt.colorbar(fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=self.font_config.font_size + 2,
                   fontweight=self.font_config.font_weight)
        plt.yticks(tick_marks, class_names, fontsize=self.font_config.font_size + 2,
                   fontweight=self.font_config.font_weight)

        # Annotate values on confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         verticalalignment="center",
                         fontsize=self.font_config.font_size + 2, fontweight=self.font_config.font_weight,
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True Label', fontsize=self.font_config.font_size + 4, fontweight=self.font_config.font_weight)
        plt.xlabel('Predicted Label', fontsize=self.font_config.font_size + 4, fontweight=self.font_config.font_weight)
        plt.savefig(os.path.join(self.result_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("Confusion matrix generated")

    def print_classification_report(self):
        """Print classification report"""
        if self.y_true is None or self.y_score is None:
            print("Please perform evaluation first")
            return

        y_pred = np.argmax(self.y_score, axis=1)
        class_names = list(self.labels_map.values())

        print("\nClassification Report:")
        print(classification_report(self.y_true, y_pred, target_names=class_names, digits=4))


# Main Program
if __name__ == "__main__":
    # Configure font
    font_config = FontConfig(font_size=35, font_weight='bold')  # Adjust font size and weight as needed

    # Load test dataset
    try:
        test_dataset = ImageDataset(r'E:\WSI-HSfeature\CRC-VAL-HE-7K', transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(f"Test dataset loaded successfully, total {len(test_dataset)} samples")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        import sys
        sys.exit(1)

    # Load model
    model = models.resnet50(weights=None)
    # Modify the last layer to match our number of classes
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 9)
    )

    try:
        model.load_state_dict(torch.load('E:\WSI-HSfeature/resnet50_weights.pth'))
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    model = model.to(device)
    model.eval()

    # Create evaluator and perform evaluation
    evaluator = ModelEvaluator(model, test_dataloader, result_dir, font_config)
    evaluator.evaluate()

    # Generate only ROC curves
    evaluator.plot_roc_curves()

    # Comment out unnecessary function calls
    # evaluator.plot_confusion_matrix()
    # evaluator.print_classification_report()

    print(f"ROC curves saved to {result_dir} folder")

# Performance Bar Chart
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
from torchvision import transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image
from matplotlib.ticker import MaxNLocator

# Set font to Arial and bold
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.weight"] = "bold"

# Create directory for saving results
result_dir = 'E:/WSI-HSfeature/9_class_bar_chart'
os.makedirs(result_dir, exist_ok=True)

# Use specified color scheme
metric_color_map = {
    "Precision": "#2171A9",  # Specified blue
    "Recall": "#EF7B20",     # Specified orange
    "F1-Score": "#329939",  # Specified green
}

# Class label mapping
class_labels = {0: "ADI", 1: "BACK", 2: "DEB", 3: "LYM", 4: "MUC",
                5: "MUS", 6: "NORM", 7: "STR", 8: "TUM"}


# Font Configuration Class
class FontConfig:
    def __init__(self, label_size=28, tick_size=24, legend_size=24, annotation_size=22):
        self.label_size = label_size  # Axis label size
        self.tick_size = tick_size    # Tick label size
        self.legend_size = legend_size  # Legend size
        self.annotation_size = annotation_size  # Annotation size


# Define Dataset Class
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        imagelist = []
        labelist = []
        # Check if img_dir exists
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Directory not found: {img_dir}")

        # Traverse all subdirectories
        for folder in os.listdir(img_dir):
            folder_path = os.path.join(img_dir, folder)
            # Only process directories
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    # Ensure processing files not subdirectories
                    if os.path.isfile(file_path):
                        imagelist.append(file)
                        labelist.append(folder)

        self.imagelist = imagelist
        self.labelist = labelist
        print(f"Loaded {len(imagelist)} images, divided into {len(set(labelist))} categories")

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labelist[idx], self.imagelist[idx])
        # Check if file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path)
        labels_map = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}
        label = labels_map[self.labelist[idx]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Evaluation Class
class ModelEvaluator:
    def __init__(self, model, test_dataloader, result_dir, font_config=None):
        self.model = model
        self.test_dataloader = test_dataloader
        self.result_dir = result_dir
        self.font_config = font_config or FontConfig()
        self.y_true = None
        self.y_score = None
        self.y_pred = None

    def evaluate(self, save_predictions=True):
        """Perform model evaluation and save results"""
        # Get model predictions
        self.y_true = []
        self.y_score = []

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                # Use softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                self.y_true.extend(labels.cpu().numpy())
                self.y_score.extend(probs.cpu().numpy())

        self.y_true = np.array(self.y_true)
        self.y_score = np.array(self.y_score)
        self.y_pred = np.argmax(self.y_score, axis=1)  # Calculate predicted labels

        # Save prediction results
        if save_predictions:
            np.save(os.path.join(self.result_dir, 'y_true.npy'), self.y_true)
            np.save(os.path.join(self.result_dir, 'y_score.npy'), self.y_score)
            np.save(os.path.join(self.result_dir, 'y_pred.npy'), self.y_pred)
            print(f"Prediction results saved to {self.result_dir}")

        return self.y_true, self.y_score, self.y_pred

    def calculate_performance_metrics(self):
        """Calculate performance metrics for each class"""
        if self.y_true is None or self.y_pred is None:
            print("Please run evaluation first")
            return None, None, None, None

        # Calculate precision, recall and F1 score for each class
        precision = precision_score(self.y_true, self.y_pred, average=None)
        recall = recall_score(self.y_true, self.y_pred, average=None)
        f1 = f1_score(self.y_true, self.y_pred, average=None)
        accuracy = accuracy_score(self.y_true, self.y_pred)

        return precision, recall, f1, accuracy

    def plot_performance_bar_chart(self):
        """Plot performance metrics bar chart"""
        if self.y_true is None or self.y_pred is None:
            print("Please run evaluation first")
            return

        # Calculate performance metrics
        precision, recall, f1, accuracy = self.calculate_performance_metrics()
        if precision is None:
            return

        class_names = [class_labels[i] for i in range(9)]

        # Set figure size (aspect ratio 24:12)
        plt.figure(figsize=(24, 12))
        plt.rcParams['figure.dpi'] = 300

        # Set bar width
        bar_width = 0.22
        index = np.arange(len(class_names))

        # Plot precision, recall and F1 score bar charts
        bar1 = plt.bar(index - bar_width, precision, bar_width, label='Precision',
                       color=metric_color_map["Precision"],
                       edgecolor='black', alpha=0.9, zorder=3)

        bar2 = plt.bar(index, recall, bar_width, label='Recall',
                       color=metric_color_map["Recall"],
                       edgecolor='black', alpha=0.9, zorder=3)

        bar3 = plt.bar(index + bar_width, f1, bar_width, label='F1-Score',
                       color=metric_color_map["F1-Score"],
                       edgecolor='black', alpha=0.9, zorder=3)

        # Add axis labels
        plt.xlabel('Class labels', fontsize=self.font_config.label_size, fontweight='bold')
        plt.ylabel('Scores', fontsize=self.font_config.label_size, fontweight='bold')
        plt.ylim(0, 1.05)  # Set y-axis range

        # Set x-axis ticks and labels
        plt.xticks(index, class_names, rotation=0, ha='center', fontsize=self.font_config.tick_size)

        # Set y-axis ticks to decimal format
        plt.gca().yaxis.set_major_locator(MaxNLocator(prune='upper', nbins=6))

        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

        # Add legend (placed above the chart)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   fancybox=True, shadow=True, ncol=3, framealpha=0.9,
                   prop={'weight': 'bold', 'size': self.font_config.legend_size})

        # Annotate values on bar charts (keep 2 decimal places)
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.2f}', ha='center', va='bottom',
                         fontsize=self.font_config.annotation_size, fontweight='bold')

        add_labels(bar1)
        add_labels(bar2)
        add_labels(bar3)

        # Add overall accuracy
        plt.axhline(y=accuracy, color='r', linestyle='--', linewidth=2.5,
                    alpha=0.7, zorder=2)

        # Add accuracy annotation
        plt.text(len(class_names) / 2, accuracy + 0.025,
                 f'Overall Accuracy: {accuracy:.2f}', color='r',
                 fontsize=self.font_config.legend_size, fontweight='bold',
                 ha='center', bbox=dict(facecolor='white', alpha=0.9, pad=6))

        # Optimize border display
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save as PNG and PDF formats
        plt.savefig(os.path.join(self.result_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.result_dir, 'performance_metrics.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Performance metrics chart saved to {result_dir} (PNG and PDF)")

    def print_classification_report(self):
        """Print classification report"""
        if self.y_true is None or self.y_pred is None:
            print("Please run evaluation first")
            return

        class_names = [class_labels[i] for i in range(9)]

        print("\nClassification Report:")
        print(classification_report(self.y_true, self.y_pred, target_names=class_names, digits=4))


# Main Program
if __name__ == "__main__":
    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Configure font
    font_config = FontConfig()

    # Load test dataset
    try:
        test_dataset = ImageDataset(r'E:\WSI-HSfeature\CRC-VAL-HE-7K', transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(f"Test dataset loaded successfully, {len(test_dataset)} samples in total")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        import sys
        sys.exit(1)

    # Load model
    model = models.resnet50(weights=None)
    # Modify the last layer to match our number of classes
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 9)
    )

    try:
        model.load_state_dict(torch.load('E:\WSI-HSfeature/resnet50_weights.pth'))
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    model = model.to(device)
    model.eval()

    # Create evaluator and perform evaluation
    evaluator = ModelEvaluator(model, test_dataloader, result_dir, font_config)
    evaluator.evaluate()

    # Generate performance metrics bar chart
    evaluator.plot_performance_bar_chart()

    # Print classification report
    evaluator.print_classification_report()

    print(f"Performance evaluation results saved to {result_dir}")
