This project is organized into four main directories to separate source code, data, experimental records, and final reports. The structure ensures a clean workflow from data processing to model evaluation.
2. Directory Structure
📂 codes/

This folder contains all the source code required to run the project.

    Scripts: Contains the main executable scripts for training and testing.
    Models: Implementation of the model architectures.
    Utils: Helper functions for data loading and preprocessing.
    Configuration: Settings and hyperparameters.

📂 data/

This folder allows for the storage and management of datasets.

    Raw Data: Place original, unmodified datasets here.
    Processed Data: Stores data after cleaning and preprocessing steps.
    Note: Ensure your dataset is formatted correctly before running the scripts in the codes/ directory.

📂 exp/ (Experiments)

This directory is used to store the outputs of your experiments.

    Logs: Training logs and runtime metrics.
    Checkpoints: Saved model weights and states during training.
    Config Dumps: Copies of the configuration files used for each specific run to ensure reproducibility.

📂 reports/

This folder contains the final analysis and visualization of the results.

    Figures: Generated plots and graphs.
    Summaries: Textual summaries or CSV files containing evaluation metrics.
    Presentations: Any final report documents.

3. Getting Started
Prerequisites

Please ensure you have the necessary dependencies installed. You can install them via the requirements file located in the codes/ directory (if available):
bash

cd codes
pip install -r requirements.txt

Data Preparation

    Navigate to the data/ directory.
    Place your dataset files in the appropriate subdirectory.

Running the Code

To start a training session or an experiment, navigate to the codes/ folder and run the main script:
bash

cd codes
# Example command
python main.py --data_dir ../data --exp_dir ../exp

Viewing Results

    Check exp/ for real-time logs and saved models.
    Once the experiment is complete, check reports/ for the final evaluation results.

4. License

This project is licensed under the [MIT / Apache 2.0] License.
