# load libraries
import pandas as pd
import nltk
import openai
import time
from openai import InvalidRequestError


nltk.download('punkt')


openai.api_key = ""

# load dataset
test_data = pd.read_csv('test_set.csv', sep=',')

start_index = 208

text_list = test_data['text'].to_list()

sublist = text_list[start_index:]

arr=[]

def perform_binary_classification(text):
    # Define your classification prompt
    prompt = f"""
          Think you are a student and you are supporting to identify self reporting social anxiety in my friends as a game for school.
          -  Decide whether the post can identify contents relevant to self reporting social anxiety or not
          -  If the post identifies self reporting social anxiety, just reply : 1
          -  If the post not identifies self reporting social anxiety, just reply : 0
            You do not have to provide explanations.
            post : ```{text}```
            """
    # Generate the classification using OpenAI GPT-3.5
    # response = openai.ChatCompletion.create(
    #     engine='gpt-3.5-turbo',
    #     prompt=prompt,
    #     max_tokens=1,  # Output only a single token ('Positive' or 'Negative')
    #     temperature=0.0,  # Keep the output deterministic
    #     n=1,  # Generate a single completion
    #     stop=None,  # Let the model decide when to stop
    # )

    response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

    # Extract the generated classification label
    label = response.choices[0]['message']['content']
    # print(response.choices[0])

    return label


final_predictions = []
# Example usage
# input_text = "Man after reading your comment I have to say I'm a little jealous, you sound like the person I want to be someday. I'm making progress and a lot of it has been on my own but the therapy that I've been getting has been helpful too. I'm realizing the progress I've made makes new progress much easier."
# classification_result = perform_binary_classification(input_text)
with open(f'chatgpt1.txt', 'a') as f:
  for i in sublist:
    classification_result = perform_binary_classification(i)
    f.write(classification_result + '\n')
    time.sleep(30)
    final_predictions.append(classification_result)
    print("Classification result:", classification_result)

test_data['label'] = final_predictions

new_df2 = test_data[['id', 'label']].copy()
new_df2.to_csv('validgpt3.tsv', sep='\t', index=False)