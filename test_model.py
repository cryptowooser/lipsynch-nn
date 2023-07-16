import glob
import torch
from torch.utils.data import DataLoader
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from data_classes import LipSyncNet, LipSyncDataset

#Constants
SAMPLE_RATE = 44100  # Your actual sample rate
HOP_LENGTH = 512  # Your actual hop length
TEST_SIZE = 0.01
# Set up DataLoader
audio_files = sorted(glob.glob('wavs/*.wav'))
text_files = sorted(glob.glob('texts/*.txt'))

dataset = LipSyncDataset(audio_files, text_files)

# Pair the audio and text files together
all_files = list(zip(audio_files, text_files))

# Perform the split
train_files, val_files = train_test_split(all_files, TEST_SIZE)

# Unzip the file pairs for each set
train_audio_files, train_text_files = zip(*train_files)
val_audio_files, val_text_files = zip(*val_files)

# Create the datasets
train_dataset = LipSyncDataset(train_audio_files, train_text_files)
val_dataset = LipSyncDataset(val_audio_files, val_text_files)


#Define the Collate_Function
def collate_fn(batch):
    # Separate the audio tensors and the label lists
    audios, labels = zip(*batch)

    # Pad the audio tensors and the labels
    # Note: We use 0 padding for audios and -1 for labels, assuming -1 is not a valid label
    audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=-1)

    # Return the padded audios and labels as a batch
    return audios_padded, labels_padded


BATCH_SIZE = 32

# Create the DataLoaders
train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            collate_fn=collate_fn)

# Instantiate the model
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    print("Warning: CUDA not found, using CPU mode.")

model_name = 'model_full_dataset_2layers.pth'
model_with_params = torch.load(model_name)

# Extract the parameters
input_size = model_with_params.input_size
hidden_size = model_with_params.hidden_size
output_size = model_with_params.output_size
num_layers = model_with_params.num_layers

# Create the model
model = LipSyncNet(input_size, hidden_size, output_size, num_layers)

# Load the model weights
model.load_state_dict(model_with_params.state_dict())
model = model.to(device)
# Make sure to call .eval() on the model for evaluation!

model.eval()
all_outputs = []
all_labels = []

# Get predictions for the validation dataset
with torch.no_grad():
    for batch in val_dataloader:
        audio, labels = batch
        audio = audio.to(device)
        labels = labels.to(device)
        outputs = model(audio)
        # The predicted class for each frame is the class with the highest score
        _, preds = torch.max(outputs, 2)
        all_outputs.extend(preds.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

# Exclude the predictions and labels where the labels are equal to -1 (the padding)
all_outputs = [
    pred for pred, label in zip(all_outputs, all_labels) if label != -1
]
all_labels = [label for label in all_labels if label != -1]

# Calculate the metrics
accuracy = accuracy_score(all_labels, all_outputs)
f1 = f1_score(all_labels, all_outputs,
              average='macro')  # Set the average parameter as you need
precision = precision_score(
    all_labels, all_outputs,
    average='macro')  # Set the average parameter as you need
recall = recall_score(all_labels, all_outputs,
                      average='macro')  # Set the average parameter as you need

print(
    f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}'
)

# Calculate the confusion matrix
cm = confusion_matrix(all_labels, all_outputs)

# Visualize the confusion matrix (optional)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion matrix')
plt.show()
