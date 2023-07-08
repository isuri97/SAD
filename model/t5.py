import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train-dataset.csv', sep=",")
val_df = pd.read_csv('val-dataset.csv', sep=",")

train_df.columns = ["text", "labels"]
# train_df['text'] = train_df['text'].astype(str)

train_set, validation_set = train_test_split(train_df, test_size=0.1)

# eval_df = pd.DataFrame(eval_data)
# eval_df.columns = ["prefix", "input_text", "target_text"]
# eval_df['target_text'] = eval_df['target_text'].astype(str)

test_sentences = val_df['text'].tolist()


# Configure the model
model_args = T5Args()
model_args.num_train_epochs = 100
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model = T5Model("t5", "t5-base", args=model_args)

# Train the model
model.train_model(train_df, eval_data=validation_set)

# Evaluate the model
result = model.eval_model(validation_set)

predictions = model.predict(test_sentences)


val_df['label'] = predictions
print(val_df)

new_df2 = val_df[['id', 'label']].copy()
new_df2.to_csv('valid2.tsv', sep='\t', index=False)