# Bridging the Sustainable Development Goals: A Multi-Label Text Classification Approach
**This repository contains the official code and pre-trained model for the paper: **"Bridging the Sustainable Development Goals: A Multi-Label Text Classification Approach for Mapping and Visualizing Nexuses in Sustainability Research"**.

:arrow_right: **Paper Link:** [Link to Published Paper will be added upon publication]
This project provides a highly generalizable, fine-tuned LUKE model for classifying text into the 17 UN Sustainable Development Goals (SDGs). It is designed to help researchers, policymakers, and organizations analyze and map SDG-related content in a robust and automated way.
**---
## :rocket: Features
* **Pre-trained Model:** A ready-to-use, fine-tuned LUKE model for high-performance multi-label SDG classification.
* **Training Pipeline:** The complete code to train the model from scratch on our multi-sectoral corpus.
* **Hyperparameter Optimization:** Integrated with Optuna for systematic hyperparameter tuning.
* **Reproducibility:** A detailed guide to reconstruct the training corpus as described in our paper.
---
## :hammer_and_wrench: Installation & Setup
Follow these steps to set up the environment and get ready to run the code.
**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```
**2. Create a Python Virtual Environment**
We strongly recommend using a virtual environment to manage dependencies.
```bash
# Create the environment
python -m venv venv
# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
**3. Install Required Libraries**
Install all necessary packages using the provided `requirements.txt` file.
```bash
pip install -r requirements.txt
```
**4. Download SpaCy English Model**
The code uses a SpaCy model for some text processing tasks. Download it with the following command:
```bash
python -m spacy download en_core_web_sm
```
---
## :computer: Usage
This script has two main modes: prediction with our pre-trained model and training a new model from scratch.
### :crystal_ball: Prediction with the Pre-trained Model (Default)
This is the easiest way to get started. Our fine-tuned model weights (`best_model.pt`) will be automatically downloaded from the Hugging Face Hub.
**1. Prepare your input**
Open the `sdgs_classifier_20250218.py` file and modify the `texts_to_predict` list inside the `run_prediction_example` function with the texts you want to classify.
**2. Run the Script**
Simply execute the Python script from your terminal:
```bash
python sdgs_classifier_20250218.py
```
The script will download the model, run the predictions, and print the detected SDGs for each text to the console.
###  :weight_lifting_woman: Training a New Model
To reproduce our training results or train the model on your own data, follow these steps.
**1. Prepare the Training Data**
The training corpus used in our paper is not included in this repository due to the terms of use of the original data sources.
To train the model, you must first reconstruct the corpus by following the detailed protocol provided in **Supplementary Information S1** of our paper. This process should result in a pandas DataFrame saved as a pickle file at the following location: `data/corpus.pkl`.
**2. Modify the Script for Training**
Open `sdgs_classifier_20250218.py` and at the very bottom of the file, uncomment the line to run the training pipeline:
```python
if __name__ == '__main__':
    # By default, this script runs a simple prediction example.
    # run_prediction_example()
    # To run the full training pipeline, uncomment the line below.
    # WARNING: This requires the pre-compiled training corpus and will take a long time.
    run_training_pipeline()
```
**3. Run the Training**
Execute the script from your terminal. The process will train the model, validate it, and save the final `best_model.pt` in a new timestamped folder inside the `results` directory.
```bash
python sdgs_classifier_20250218.py
```
---
## :scroll: Citation
If you use this code or our model in your research, please cite our paper:
```bibtex
@article{Miyashita2025,
  author    = {Naoto Miyashita and Takanori Matsui and Chihiro Haga and Naoki Masuhara and Shun Kawakubo},
  title     = {Bridging the Sustainable Development Goals: A Multi-Label Text Classification Approach for Mapping and Visualizing Nexuses in Sustainability Research},
  journal   = {Sustainability Science},
  year      = {2025},
  % Add Volume, Pages, DOI upon publication
}
```
---
## :page_facing_up: License
This project is licensed under the [Apache 2.0](LICENSE.md).
---
## :e-mail: Contact
For any questions, please contact the corresponding author:
**Takanori Matsui** (matsui@see.eng.osaka-u.ac.jp)
