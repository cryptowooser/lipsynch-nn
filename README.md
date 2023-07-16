# Rhubarb Lip Sync - Neural Network Implementation

This is a Python neural network implementation of [Rhubarb Lip Sync](https://github.com/DanielSWolf/rhubarb-lip-sync), designed to enable complex lip movement for real-time chatbots.

This project utilizes a simple neural network trained on pairs of spoken texts and Rhubarb Lip Sync outputs, approximating lip movements with around 75% accuracy. This level of accuracy is sufficient for generating a sense of realism in most applications.

Please note, if your application does not require real-time performance, you might want to consider using Rhubarb Lip Sync directly.

## How to Use
For now, only inference is supported, as the training code is being heavily refactored. If for some reason you need to train your own model and can't wait, let me know. 
For inference, use the following command:
```
python .\inference.py --wav_file_name .\001.wav --model_name model_full_dataset_2layers.pth  
```


## To-Do List
1. Finish refactoring training code and helper classes.
2. Convert from using .pth to using SafeTensors.
3. Add video example to README.md.
4. Build a proper requirements file. 


## Current Status

The code is currently undergoing refactoring and users may encounter errors, particularly when attempting to train their own models. However, the provided model (`model_full_dataset_2layers.pth`) should be satisfactory for most purposes. It's been trained on over 80 GB of WAV files from a variety of sources, providing a comprehensive and versatile foundation for lip-syncing tasks.

## License and Use

This code is available under the MIT license and is free for anyone to use without obligation. However, I would be delighted if you'd drop me a line to let me know if and how you're using it! 

## Contributions and Feedback

Please feel free to contribute to this project or provide feedback by opening an issue or pull request on GitHub. Your insights are greatly appreciated!
