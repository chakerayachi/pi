import streamlit as st
import torch
import torch.nn as nn
import numpy as np 
import json
from transformers import AutoTokenizer, AutoModel
# Define the TransE model
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Create a default tensor for unknown entities
        self.default_entity_embedding = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, sample):
        subject_idx, relation_idx, object_idx = sample[:, 0], sample[:, 1], sample[:, 2]

        # Embedding lookup for known entities
        subject = self.entity_embeddings(subject_idx)
        relation = self.relation_embeddings(relation_idx)
        object_ = self.entity_embeddings(object_idx)

        # Embedding lookup for unknown entities
        unknown_entities_mask = (subject_idx >= self.num_entities) | (object_idx >= self.num_entities)
        unknown_entities_idx = subject_idx[unknown_entities_mask]  # or use object_idx
        unknown_entities = self.default_entity_embedding.expand(len(unknown_entities_idx), -1)

        # Replace embeddings for unknown entities
        subject[unknown_entities_mask] = unknown_entities
        object_[unknown_entities_mask] = unknown_entities

        return subject, relation, object_
state_dict = torch.load("model1.h5")

# Set the number of entities and relations based on the loaded state_dict
num_entities = state_dict['entity_embeddings.weight'].size(0)
num_relations = state_dict['relation_embeddings.weight'].size(0)

# Load the entity2idx dictionary from the file
with open("entity2idx.json", "r") as file:
    entity2idx = json.load(file)

# Set the embedding dimension
embedding_dim = state_dict['entity_embeddings.weight'].size(1)

# Create the model with the correct number of entities and relations
model = TransE(num_entities, num_relations, embedding_dim)

# Now load the state_dict
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()
# Load entities from a file
with open("entities.txt", "r") as file:
    entities = [line.strip() for line in file]
# Load relations from a file
with open("relations.txt", "r") as file:
    relations = [line.strip() for line in file]

class BertEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BertEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Initialize the BertEmbedding module
bert_embedding = BertEmbedding(768, embedding_dim)

def recommend_related_entities(transE_model, bert_model, bert_embedding_module, entity, entity2idx, entities, tokenizer, k=5):
    # Set the model to evaluation mode
    transE_model.eval()

    # Check if the entity is in the entity2idx dictionary
    if entity in entity2idx:
        # Get the embedding of the given entity using TransE
        entity_idx = entity2idx[entity]
        entity_embedding = transE_model.entity_embeddings(torch.tensor([entity_idx]))
    else:
        # Use a language model to generate an embedding for the entity description
        inputs = tokenizer(entity, return_tensors='pt')
        outputs = bert_model(**inputs)
        bert_embedding_tensor = outputs.last_hidden_state.mean(dim=1)
        # Reduce the dimensionality of the Bert embedding
        entity_embedding = bert_embedding_module(bert_embedding_tensor)

    # Calculate the distance between the given entity and all other entities
    distances = torch.norm(transE_model.entity_embeddings.weight - entity_embedding, dim=1)

    # Get the indices of the top k closest entities
    _, indices = torch.topk(distances, k, largest=False)

    # Convert the indices to entity names
    recommendations = [entities[idx] for idx in indices]

    return recommendations
# Define the prediction function
def predict_relation(transE_model, bert_model, bert_embedding_module, entity1, entity2, entity2idx, tokenizer):
    transE_model.eval()
    bert_model.eval()

    # Check if the entities are in the entity2idx dictionary
    if entity1 in entity2idx:
        entity1_idx = entity2idx[entity1]
        entity1_embedding = transE_model.entity_embeddings(torch.tensor([entity1_idx]))
    else:
        # Use a language model to generate an embedding for the entity description
        inputs = tokenizer(entity1, return_tensors='pt')
        outputs = bert_model(**inputs)
        bert_embedding_tensor = outputs.last_hidden_state.mean(dim=1)
        # Reduce the dimensionality of the Bert embedding
        entity1_embedding = bert_embedding_module(bert_embedding_tensor)

    if entity2 in entity2idx:
        entity2_idx = entity2idx[entity2]
        entity2_embedding = transE_model.entity_embeddings(torch.tensor([entity2_idx]))
    else:
        # Use a language model to generate an embedding for the entity description
        inputs = tokenizer(entity2, return_tensors='pt')
        outputs = bert_model(**inputs)
        bert_embedding_tensor = outputs.last_hidden_state.mean(dim=1)
        # Reduce the dimensionality of the Bert embedding
        entity2_embedding = bert_embedding_module(bert_embedding_tensor)

    diff = entity2_embedding - entity1_embedding

    distances = torch.norm(transE_model.relation_embeddings.weight - diff, dim=1)

    relation_idx = torch.argmin(distances).item()

    relation = relations[relation_idx]

    return relation

# Create a Streamlit app
# st.title('Knowledge Graph Entity Recommender')


# # Create a selectbox for the user to choose the operation
# operation = st.selectbox('Choose an operation:', ['Recommend related entities', 'Predict relation'])

# if operation == 'Recommend related entities':
#     entity = st.text_input('Enter an entity:')

#     if st.button('Recommend'):
#         recommendations = recommend_related_entities(model, bert_model, bert_embedding, entity, entity2idx, entities, bert_tokenizer, k=3)
#         st.write(f"Entities related to {entity}: {recommendations}")
# else:
#     entity1 = st.text_input('Enter the first entity:')
#     entity2 = st.text_input('Enter the second entity:')

#     if st.button('Predict'):
#         relation = predict_relation(model, bert_model, bert_embedding, entity1, entity2, entity2idx, bert_tokenizer)
#         st.write(f"The most likely relation between {entity1} and {entity2} is {relation}")

    # Load external CSS file

st.markdown(
        """
        <style>

        /* Add your CSS styles here */
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            background-image: url('/back.png'); /* Replace 'background_image.jpg' with your image file */
            background-size: cover;
            background-repeat: no-repeat;
            color: #333;
            margin: 0;
            padding: 20px;
        }
        .stApp {
            max-width: 800px;
            margin: auto;
        }
        .stButton button {
            background-color: #008CBA;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .stButton button:hover {
            background-color: #005f73;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
st.title('Knowledge Graph Entity Recommender')

    # Create a selectbox for the user to choose the operation
operation = st.selectbox('Choose an operation:', ['Recommend related entities', 'Predict relation'])

if operation == 'Recommend related entities':
    entity = st.text_input('Enter an entity:', help='Type an entity to find related entities')

    if st.button('Recommend'):
        if entity.strip():  # Check if the input entity is not empty
            recommendations = recommend_related_entities(model, bert_model, bert_embedding, entity, entity2idx, entities, bert_tokenizer, k=3)
            st.write(f"Entities related to {entity}: {recommendations}")
        else:
            st.warning('Please enter an entity.')

else:
    entity1 = st.text_input('Enter the first entity:', help='Type the first entity')
    entity2 = st.text_input('Enter the second entity:', help='Type the second entity')

    if st.button('Predict'):
        if entity1.strip() and entity2.strip():  # Check if both input entities are not empty
            relation = predict_relation(model, bert_model, bert_embedding, entity1, entity2, entity2idx, bert_tokenizer)
            st.write(f"The most likely relation between {entity1} and {entity2} is {relation}")
        else:
            st.warning('Please enter both entities.')

