# Course-AI-Design-Project

## Source Code Overview
This directory contains the source code for training a language model. The code is organized to allow flexibility in choosing datasets, model architectures, and training methods. The goal is to create a model that can generate 10 different coherent and unambiguous sentences.

## Directory Structure
- `data/`: Contains scripts for data preprocessing and dataset management.
- `models/`: Includes model architecture definitions and utilities.
- `training/`: Contains the training scripts and configurations.
- `utils/`: Utility functions and helper scripts.
- `generate_sentences.py`: Script to generate sentences using the trained model.
- `train_model.py`: Main script to train the language model.

## Requirements
- Ensure all dependencies are installed. Refer to `requirements.txt` for the list of required packages.
- The model should generate 10 different coherent and unambiguous sentences.

## Running the Code
1. **Dataset Preparation**: Place your dataset in the `data/` directory and preprocess it using the provided scripts.
2. **Model Training**: Use `train_model.py` to train your model. Adjust configurations as needed.
3. **Sentence Generation**: Use `generate_sentences.py` to generate sentences with the trained model.

## Example Usage
```bash
# Navigate to the src directory
cd /Users/maple_of_ten/Project/Course-AI-Design-Project/src

# Run the training script
python train_model.py

# Generate sentences
python generate_sentences.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments
- Thanks to the open-source community for providing valuable resources and tools.
- Special thanks to the course instructors for their guidance.

## Contact
For any questions or issues, please open an issue in the repository or contact the project maintainer at your.email@example.com.

