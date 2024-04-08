import subprocess
import torch
import pandas as pd
import numpy as np
import os
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from autoPyTorch.api.NLP_classification import TabularClassificationTask, text_embeddings
from codes.automated_fine_tuning_BERT_tabular_classification import rename_columns_based_on_content

# Define the paths to your CSV file and Python script
#csv_path = "/home/safikhani/main_repository/Auto-PyTorch_autoNLP/data/500emotions.csv"
#fine_tuning_step = "/home/safikhani/main_repository/Auto-PyTorch_autoNLP/codes/automated_fine_tuning_BERT_tabular_classification.py"

# Execute the script with the CSV file as an argument
#try:
#    result = subprocess.run(['python', fine_tuning_step, csv_path], check=True, text=True, capture_output=True)
#    print("Script executed successfully.")
#    print("Output:", result.stdout)
#except subprocess.CalledProcessError as e:
#    print("Error occurred while executing the script.")
#    print("Error:", e)
#    print("Output:", e.output)

# csv_path = "/home/safikhani/main_repository/Auto-PyTorch_autoNLP/data/500emotions.csv"
# # test_data_name = '500emotions.csv'
# fine_tuning_step = "/home/safikhani/main_repository/Auto-PyTorch_autoNLP/codes/automated_fine_tuning_BERT_tabular_classification.py"

# # # Skript mit Daten als Argument ausführen
# subprocess.run(['python', fine_tuning_step, csv_path])

# try:
#         data = rename_columns_based_on_content(csv_path )
#         print(data.head())
# except Exception as e:
#         print(f"An error occurred: {e}")

#     # Determine the number of unique classes
# num_labels = data['label'].nunique()
# # Assuming 'data' is your DataFrame and it has a 'label' column
# unique_labels = sorted(data['label'].unique())
# label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
# data['label'] = data['label'].map(label_mapping)

# # Now, ensure num_labels is correctly set
# num_labels = len(unique_labels)


# # Prepare data (Assumption: 'text' is the column with input texts)
# data = data.dropna(subset=['text', 'label']).assign(text=lambda x: x['text'].astype(str))

# # Output file path dynamically includes the test data name
# output_csv_path = os.path.join("/home/safikhani/main_repository/Auto-PyTorch_autoNLP/embeddings", f"{os.path.splitext(test_data_name)[0]}_embeddings.csv")

# # Function to generate embeddings
# def generate_embeddings(texts, tokenizer, model, max_length=512):
#     embeddings = []
#     for text in texts:
#         inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")
#         inputs = {key: value.to('cuda') for key, value in inputs.items()}  # Move inputs to GPU
        
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
#         embeddings.append(embedding)
    
#     return embeddings

# # Load data and initialize tokenizer and model
# model_name = "/home/safikhani/main_repository/Auto-PyTorch_autoNLP/Fine-tuned-models/500emotions"
# data = pd.read_csv(csv_path)
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
# model.to('cuda')  # Move model to GPU

# if not os.path.exists(output_csv_path):
#     feature_col = 'text'
#     texts = data[feature_col].tolist()
#     embeddings = generate_embeddings(texts, tokenizer, model)
    
#     df_embeddings = pd.DataFrame(embeddings)
#     df_embeddings['label'] = data['label']
#     df_embeddings.to_csv(output_csv_path, index=False)
# else:
#     df_embeddings = pd.read_csv(output_csv_path)
#     print("Embeddings already exist. Skipping generation.")

# # Separate features and target variable
# X = df_embeddings.drop(['label'], axis=1)
# y = df_embeddings['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# api = TabularClassificationTask()

# api.search(
#     X_train=X_train,
#     y_train=y_train,
#     X_test=X_test,
#     y_test=y_test,
#     optimize_metric='accuracy',
#     total_walltime_limit=300,
#     func_eval_time_limit_secs=50
# )

# y_pred = api.predict(X_test)
# score = api.score(y_pred, y_test)
# print("Accuracy score", score)
