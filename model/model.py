from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import logging


# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

# Load data
from print_stat import print_information

train_df = pd.read_csv('train-dataset.csv', sep=",")
# val_df = pd.read_csv('val-dataset.csv')

train_df = train_df[['id', 'text', 'labels']]
# print(train_df)

# Optional model configuration
train_set, validation_set = train_test_split(train_df, test_size=0.2)
new_df = validation_set[['id', 'labels']].copy()
new_df.to_csv('test.tsv', index=False)

print(validation_set)
test_sentences = validation_set['text'].tolist()

# define hyperparameter
train_args = {"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": 4,
             "train_batch_size": 8,
             "use_multiprocessing": False,
             "use_multiprocessing_for_evaluation":False,
             "n_fold":1
        }

# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-cased",
    args=train_args,
)

# Train the model
model.train_model(train_set)

# Make predictions with the model
predictions, raw_outputs = model.predict(test_sentences)
print(predictions)

validation_set['prediction'] = predictions
print(validation_set)

new_df2 = validation_set[['id', 'prediction']].copy()
new_df2.to_csv('valid.tsv', index=False)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(validation_set)



