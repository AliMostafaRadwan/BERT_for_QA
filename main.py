import transformers
import torch

# Preprocess the book data

def preprocess_book(book):
    # Split the book into individual sentences
    sentences = book.split('.')

    # Tokenize the sentences
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
    tokenized_sentences = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences]

    # Tokenize the sentences
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
    tokenized_sentences = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences]

    # Convert the tokenized sentences to tensors
    tensorized_sentences = [torch.tensor(tokenized_sentence) for tokenized_sentence in tokenized_sentences]

    # Return the tensorized sentences
    return tensorized_sentences


book = open('Atomic Habits.txt', 'r').read()



tensorized_book = preprocess_book(book)

# Split the book data into training and evaluation sets
train_data = tensorized_book[:int(len(tensorized_book) * 0.8)]
eval_data = tensorized_book[int(len(tensorized_book) * 0.8):]

# Create the BERT question answering model
model = transformers.BertForQuestionAnswering.from_pretrained('bert-base-cased')

# Train the model
model.fit(train_data, epochs=5, batch_size=32)

# Evaluate the model
model.evaluate(eval_data)

# Fine-tune the model (optional)
