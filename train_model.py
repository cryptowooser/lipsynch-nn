import glob
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from funrun import create_run_name
from data_classes import LipSyncNet, LipSyncDataset

#Constants
SAMPLE_RATE = 44100
HOP_LENGTH = 512
BATCH_SIZE = 32  # You can adjust this value based on your system's memory
NUM_EPOCHS = 20
INPUT_SIZE = 13
HIDDEN_SIZE = 256
OUTPUT_SIZE = 7
NUM_LAYERS = 2


def collate_fn(batch):
    """
    Collate function to pad audio tensors and label lists in a batch.

    This function is used as the `collate_fn` argument in a PyTorch DataLoader. It takes a batch of data and 
    applies padding to make all sequences in the batch the same length. For audios, 0 padding is used, and for 
    labels, -1 padding is used (assuming -1 is not a valid label).

    Parameters
    ----------
    batch : list
        A list of tuples, where each tuple contains an audio tensor and a corresponding label list.

    Returns
    -------
    tuple
        A tuple containing the padded audios tensor and the padded labels tensor. Both tensors have the same sequence
        length, which is equal to the length of the longest sequence in the original batch.
    """
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


#--------- MAIN FUNCTION ---------
def main():
    run_name = create_run_name('english-nouns.txt', 'english-adjectives.txt')
    writer = SummaryWriter(f'runs/{run_name}')
    # Set up DataLoader
    audio_files = sorted(glob.glob('wavs/*.wav'))
    text_files = sorted(glob.glob('texts/*.txt'))

    # Pair the audio and text files together
    all_files = list(zip(audio_files, text_files))

    # Perform the split
    train_files, val_files = train_test_split(all_files, test_size=0.2)
    # Unzip the file pairs for each set
    train_audio_files, train_text_files = zip(*train_files)
    val_audio_files, val_text_files = zip(*val_files)

    # Create the datasets
    train_dataset = LipSyncDataset(train_audio_files, train_text_files)
    val_dataset = LipSyncDataset(val_audio_files, val_text_files)


    # Create the DataLoaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=4,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                collate_fn=collate_fn,
                                num_workers=4,
                                pin_memory=True)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        print("Warning: CUDA not found, using CPU mode.")

    # Create the model, loss function and optimizer
    model = LipSyncNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
                       NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    #AMP Optimizations
    from torch.cuda.amp import autocast, GradScaler
    scaler = torch.cuda.amp.GradScaler()

    # Number of epochs to train for

    print("Beginning Training")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for i, batch in enumerate(train_dataloader):
            # Get the input and target
            audio, labels = batch
            audio = audio.to(device)
            labels = labels.to(device)
            # Clear the gradients
            optimizer.zero_grad()
            # Forward propagation
            with torch.cuda.amp.autocast():
                outputs = model(audio)

                # Create a mask by filtering out all labels values equal to -1 (the pad token)
                mask = (labels.view(-1) != -1).to(device)

                # We apply the mask to the outputs and labels tensors
                outputs = outputs.view(-1, OUTPUT_SIZE)[mask]
                labels = labels.view(-1)[mask]

                # Compute the loss and do backprop
                loss = criterion(outputs, labels)

            #Gradscaler Phase
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate the loss
            scheduler.step()

            train_loss += loss.item()

            writer.add_scalar('Training Loss',
                              loss.item(),
                              global_step=epoch * len(train_dataloader) + i)
        # Compute the average training loss for this epoch
        train_loss /= len(train_dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                # Get the input and target
                audio, labels = batch
                audio = audio.to(device)
                labels = labels.to(device)
                # Forward propagation
                outputs = model(audio)

                # Create a mask by filtering out all labels values equal to -1 (the pad token)
                mask = (labels.view(-1) != -1).to(device)

                #Enable Autocast
                # We apply the mask to the outputs and labels tensors
                outputs = outputs.view(-1, OUTPUT_SIZE)[mask]
                labels = labels.view(-1)[mask]

                # Compute the loss
                loss = criterion(outputs, labels)
                # Accumulate the loss
                val_loss += loss.item()
                writer.add_scalar('Validation Loss',
                                  loss.item(),
                                  global_step=epoch * len(train_dataloader) +
                                  i)
        # Compute the average validation loss for this epoch
        val_loss /= len(val_dataloader)

        print(
            f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}'
        )
        model.input_size = INPUT_SIZE
        model.hidden_size = HIDDEN_SIZE
        model.output_size = OUTPUT_SIZE
        model.num_layers = NUM_LAYERS
        model.epoch = epoch + 1
        model.loss = loss

        torch.save(model, f"model_{epoch+1}.pth")

    torch.save(model.state_dict(), 'model.pth')
    writer.close()


if __name__ == '__main__':
    main()