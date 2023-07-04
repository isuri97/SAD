from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import logging


# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

# Load data
train_df = pd.read_csv('train-dataset.csv')
val_df = pd.read_csv('val-dataset.csv')

train_df = train_df[['text', 'label']]
# train_df['labels'] = encode(train_df["label"])

# train_df.columns = ["text", "labels"]
print(train_df)
# val_df = val_df[['text', 'label']]

# model_args = ClassificationArgs(num_train_epochs=1)

# Optional model configuration
train_set, validation_set = train_test_split(train_df, test_size=0.2)


# define hyperparameter
train_args = {"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": 4}

# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-cased",
    num_labels=4,
    args=train_args
)

# Train the model
model.train_model(train_set)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(validation_set)
