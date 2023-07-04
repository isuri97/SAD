from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import logging


# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

# Load data
train_df = pd.read_csv('train-dataset.csv',sep=",")
# val_df = pd.read_csv('val-dataset.csv')

train_df = train_df[['text', 'labels']]
# train_df['label'] = encode(train_df['label'])
print(train_df)

# Optional model configuration
train_set, validation_set = train_test_split(train_df, test_size=0.2)


# define hyperparameter
train_args = {"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": 4,
             "train_batch_size": 16
                           }

# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-cased",
    args=train_args,

)

# Train the model
model.train_model(train_df)
#
# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(validation_set)
