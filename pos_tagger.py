
import warnings
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import torch.nn.functional as F
import sys
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torchtext
from conllu import parse
from torch.utils.data import Dataset, DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt



p=2
s=2
model_type=sys.argv[1]
print(model_type)
# from conllu import parse

"""set gpu or cpu"""

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


"""file names"""

trainfilePath="./UD_English-Atis/en_atis-ud-train.conllu"
validationfilePath="./UD_English-Atis/en_atis-ud-dev.conllu"
testfilePayh="./UD_English-Atis/en_atis-ud-test.conllu"


"""Load data"""

warnings.filterwarnings("ignore")
torch.manual_seed(1)
with open(trainfilePath) as f:
    train_data = parse(f.read())
with open(validationfilePath) as f:
    dev_data = parse(f.read())
with open(testfilePayh) as f:
    test_data = parse(f.read())

"""Prepare data"""

# train_data
def prepare_datasets(dataset):
    # dict1={}
    mod_data = []
    for idx in range(len(dataset)):
        tempword = []
        temptag = []
        for jdx in range(len(dataset[idx])):
            tempword.append(dataset[idx][jdx]["form"])
            temptag.append(dataset[idx][jdx]["upos"])



        mod_data.append([tempword, temptag])
    return mod_data
mod_train_data = prepare_datasets(train_data)
mod_dev_data = prepare_datasets(dev_data)
mod_test_data = prepare_datasets(test_data)

"""Number of sentense in each catogery"""

print(f"Number of training examples: {len(mod_train_data)}")
print(f"Number of validation examples: {len(mod_dev_data)}")
print(f"Number of testing examples: {len(mod_test_data)}")

"""create vocab for train data"""

words_list = [sublist[0] for sublist in mod_train_data]
word_vocab = torchtext.vocab.build_vocab_from_iterator(words_list, min_freq=2)
word_vocab.insert_token("<unk>", 0)
word_vocab.set_default_index(word_vocab["<unk>"])

tags_list = [sublist[1] for sublist in mod_train_data]
tag_vocab = torchtext.vocab.build_vocab_from_iterator(tags_list)

itos_mapping = tag_vocab.get_stoi()

# print(itos_mapping)
# print(f"Unique words: {len(word_vocab)}")
# print(f"Unique tags: {len(tag_vocab)}")



X_train = [sublist[0] for sublist in mod_train_data]
Y_train=[sublist[1] for sublist in mod_train_data]


X_valid = [sublist[0] for sublist in mod_dev_data]
Y_valid=[sublist[1] for sublist in mod_dev_data]



X_test = [sublist[0] for sublist in mod_test_data]
Y_test=[sublist[1] for sublist in mod_test_data]


def padding(X,start_symbol,end_symbol,p,s):
  modified_list_of_lists = [[start_symbol] * p + sublist + [end_symbol] * s for sublist in X]
  return modified_list_of_lists



def replace_word_in_lists(list_of_lists, X_list_of_list, word_to_replace):
    updated_list_of_lists = []
    updated_X_list_of_list = []

    for sub_list, X_sub_list in zip(list_of_lists, X_list_of_list):
        updated_sub_list = []
        updated_sub_list_X = []

        for word, X_word in zip(sub_list, X_sub_list):
            if word == word_to_replace:
                # Skip adding word_to_replace and its corresponding element
                continue

            updated_sub_list.append(word)
            updated_sub_list_X.append(X_word)

        updated_list_of_lists.append(updated_sub_list)
        updated_X_list_of_list.append(updated_sub_list_X)

    return updated_list_of_lists, updated_X_list_of_list






def sequence_to_idx(X, ix):
  encoded_X=[]
  for sent in X:
    ans=[]


    for word in sent:
      try:
        ans.append(ix[word])
      except KeyError:
        ans.append(0)

    encoded_X.append(ans)
  return encoded_X




def sequence_to_idx_user(X, ix):
  encoded_X=[]

  for word in X:
    try:
      encoded_X.append(ix[word])
    except KeyError:
      encoded_X.append(0)
  return encoded_X




def one_hot_encodeing_ffnn(X,vector_size):
  one_hot_encoding_vec=[]
  for vec in X:
    one_hot_of_sentense=[]
    for idx in vec:
      vector = [1 if i == idx else 0 for i in range(vector_size)]
      one_hot_of_sentense.append(vector)
    one_hot_encoding_vec.append(one_hot_of_sentense)
  return one_hot_encoding_vec

def one_hot_encodeing_user_ffnn(X,vector_size):
  one_hot_encoding_vec=[]

  for idx in X:
    vector = [1 if i == idx else 0 for i in range(vector_size)]
    one_hot_encoding_vec.append(vector)
  return one_hot_encoding_vec


def create_matrix_ffnn(X, Y,p,s):
    X_mat = []
    Y_mat = []
    for x, y in zip(X, Y):
        sublists = [x[i:i+p+s+1] for i in range(len(x) - (p+s+1) + 1)]
        sublist2 = [y[i] for i in range(len(y))]  # Adjust range for Y_mat
        X_mat.extend(sublists)
        Y_mat.extend(sublist2)
    return X_mat, Y_mat



def create_matrix_user_ffnn(X,p,s):
    X_mat = []


    sublists = [X[i:i+p+s+1] for i in range(len(X) - (p+s+1) + 1)]
    X_mat.extend(sublists)

    return X_mat


class FFNN_POS_Tagger(nn.Module):
    def __init__(self, embedding_dim, p, s, hidden_dim, output_dim):
        super(FFNN_POS_Tagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.p = p
        self.s = s
        self.input_dim = embedding_dim * (p + s + 1)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs):
        x = inputs.view(-1, self.input_dim)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def transform_to_one_hot_ffnn(tensor):
    """
    Transforms each row in the given PyTorch tensor to a one-hot encoding format,
    where the maximum value in each row is set to 1 and all other values are set to 0.

    Args:
    - tensor (torch.Tensor): A PyTorch tensor containing float values.

    Returns:
    - torch.Tensor: The transformed tensor with one-hot encoding format.
    """
    transformed_tensor = torch.zeros_like(tensor)
    max_values, max_indices = torch.max(tensor, dim=1)
    for i in range(len(tensor)):
        transformed_tensor[i, max_indices[i]] = 1.
    return transformed_tensor


def transform_to_one_hot_user_ffnn(tensor):
    max_index = torch.argmax(tensor)
    return max_index

def howmany_equal_ffnn(predicted_val, targets):
    t = 0
    for t1, t2 in zip(predicted_val, targets):
        if torch.all(t1.eq(t2)):
            t += 1
    return t


def train_model_ffnn(model,criterion,optimizer,num_epochs,train_loader,validation_loader):
  epoch_v=[]
  val_v=[]
  loss_v=[]

  for epoch in range(num_epochs):
      model.train()

      for inputs, targets in train_loader:
          optimizer.zero_grad()
          inputs = inputs.float()  # Convert input to float data type
          outputs = model(inputs)
          targets = targets.float()
          loss = criterion(outputs, targets)

          loss.backward()
          optimizer.step()

      # Validation loop
      model.eval()
      with torch.no_grad():
          total_val = 0
          correct_val = 0
          loss_val=0


          for inputs, targets in validation_loader:
              inputs = inputs.float()  # Convert input to float data type
              outputs = model(inputs)
              targets = targets.float()


              predicted_val = transform_to_one_hot_ffnn(outputs)
              loss = criterion(outputs, targets)
              loss_val=loss_val+loss.item()

              total_val += targets.size(0)
              # correct_val += (predicted_val == targets).sum().item()
              correct_match= howmany_equal_ffnn(predicted_val, targets)
              correct_val=correct_val+correct_match




          accuracy_val = correct_val / total_val
          loss_val=loss_val/total_val


          epoch_v.append(epoch+1)
          val_v.append(accuracy_val)
          loss_v.append(loss_val)
          print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy_val}')
  return epoch_v,val_v,loss_v

def test_model_ffnn(model,test_loader):
  empty_shape = (0, 13)
  pred_vector= torch.empty(empty_shape)
  model.eval()
  with torch.no_grad():
      total_test = 0
      correct_test = 0

      for inputs, target in test_loader:

          inputs = inputs.float()   # Convert input to float data type
          outputs = model(inputs)


          predicted_val = transform_to_one_hot_ffnn(outputs)
          pred_vector = torch.cat((pred_vector, predicted_val), dim=0)
          total_test += 1
          correct_match= howmany_equal_ffnn(predicted_val, target)

          correct_test += correct_match
      accuracy_test = correct_test / total_test
      print(f'Test Accuracy: {accuracy_test}')
  return pred_vector

def plot_graph(epoch_v, loss_v,val_v,train_loss,train_accuracy):
  
  plt.figure(figsize=(10, 5))
  plt.plot(epoch_v, loss_v, marker='o', color='green', label='Validation Loss')
  plt.title('Validation Loss vs Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)
  plt.show()




  # Plotting accuracy vs epoch
  plt.figure(figsize=(10, 5))
  plt.plot(epoch_v, val_v, marker='o', color='blue', label='Validation Accuracy')
  plt.title('Validation Accuracy vs Epoch')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.grid(True)
  plt.show()
  
  # draw graph for train data only for LSTN

  if model_type == 'r':
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_v, train_loss, marker='o', color='green', label='Validation Loss')
    plt.title('Train Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.figure(figsize=(10, 5))
    plt.plot(epoch_v, train_accuracy, marker='o', color='blue', label='Validation Accuracy')
    plt.title('Train Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def confusion_matrix_draw(y_test_tensor,pred_vector):
  conf_matrix = confusion_matrix(y_test_tensor, pred_vector)

  plt.figure(figsize=(10, 8))
  sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.title('Confusion Matrix')
  plt.show()



  print("Classification Report :")
  report = classification_report(y_test_tensor, pred_vector, output_dict=True)
  df = pd.DataFrame(report).transpose()
  cm = sns.light_palette("green", as_cmap=True)
  styled_table = df.style.background_gradient(cmap=cm)
  return styled_table,conf_matrix


def find_tags_ffnn(data,model,tag_vocab):
  ans=[]
  model.eval()
  with torch.no_grad():

      for inputs1 in data:

          inputs1 = inputs1.float()   # Convert input to float data type
          outputs1 = model(inputs1)
          predicted_val1 = transform_to_one_hot_user_ffnn(outputs1)
          tag = next((key for key, value in tag_vocab.get_stoi().items() if value == predicted_val1), None)
          ans.append(tag)
  return ans

def run_model_on_user_input_ffnn(model,word_vocab,tag_vocab,user_input):
  print("Your Sentense:", user_input)
  input= user_input.lower().split()
  # print(input)
  input_p=["<s>"] * p + input + ["</s>"] * s

  input_encoded=sequence_to_idx_user(input_p,word_vocab)

  input_one_hot_encoded=one_hot_encodeing_user_ffnn(input_encoded,len(word_vocab))

  input_mat_test = create_matrix_user_ffnn(input_one_hot_encoded,p,s)

  user_input = np.array(input_mat_test)
  user_input_tensor = torch.tensor(user_input)

  user_input_loader = DataLoader(user_input_tensor, batch_size=1)


  ans=find_tags_ffnn(user_input_loader,model,tag_vocab)
  print("word","-->","tag")
  for word, tag in zip(input,ans):
     print(word,"-->",tag)




# for LSTM


class LSTMTagger(nn.Module):
    def __init__(self, word_embedding_dim, word_hidden_dim, vocab_size, tagset_size, dropout_fac):
        super(LSTMTagger, self).__init__()
        self.word_hidden_dim = word_hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim, word_hidden_dim, num_layers=Layer_number, bidirectional=Bidirectional)

        self.hidden2tag = nn.Linear(word_hidden_dim * 2, tagset_size)

        self.dropout = nn.Dropout(dropout_fac)

    def forward(self, sentence):
        embeds = self.dropout(self.word_embeddings(sentence))
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def calc_correct_lstm(targets_val,indices_val):
  ans=0.0
  for i,j in zip(targets_val,indices_val):
    if i==j:
      ans=ans+1.0
  ans=ans/len(indices_val)
  return ans

def train_model_lstm(model,loss_function,optimizer,num_epochs,X_train_final,Y_train_final,X_valid_final,Y_valid_final):
  train_accuracy=[]
  train_loss=[]
  valid_accuracy=[]
  valid_loss=[]
  epoch_v=[]


  for epoch in range(num_epochs):
    epoch_v.append(epoch+1)
    model.train()
    acc=0
    loss=0
    v_loss=0
    v_acc=0
    for input, targets in zip(X_train_final,Y_train_final):
      input = torch.tensor(input, dtype=torch.long)
      targets = torch.tensor(targets, dtype=torch.long)
      model.zero_grad()
      tag_scores = model(input)
      loss = loss_function(tag_scores, targets)
      loss.backward()
      optimizer.step()
      loss += loss.item()
      _, indices = torch.max(tag_scores, 1)

      # acc += torch.mean(torch.tensor(targets == indices, dtype=torch.float))
      acc = acc+calc_correct_lstm(targets,indices)

    loss = loss / len(mod_train_data)
    train_loss.append(float(loss))

    acc = acc / len(mod_train_data)
    train_accuracy.append(float(acc))

    for input_val, targets_val in zip(X_valid_final,Y_valid_final):
      input_val = torch.tensor(input_val, dtype=torch.long)
      targets_val = torch.tensor(targets_val, dtype=torch.long)
      model.zero_grad()
      tag_scores_val = model(input_val)

      loss_val = loss_function(tag_scores_val, targets_val)

      v_loss += loss_val.item()
      _, indices_val = torch.max(tag_scores_val, 1)

      # v_acc = v_acc+torch.mean(torch.tensor(targets_val == indices_val, dtype=torch.float))
      v_acc = v_acc+calc_correct_lstm(targets_val,indices_val)
    v_loss = v_loss / len(mod_dev_data)
    valid_loss.append(float(v_loss))
    v_acc = v_acc / len(mod_dev_data)
    valid_accuracy.append(float(v_acc))
    print(f"Epoch {epoch+1} \t\t Training Loss: {loss} \t\t Training Acc: {acc}")
    print(f"Epoch {epoch+1} \t\t Validation Loss: {v_loss} \t\t Validation Acc: {v_acc}")

  return train_accuracy,train_loss,valid_accuracy,valid_loss,epoch_v


def flatten_list(nested_list):
   return [item for sublist in nested_list for item in sublist]


def test_model_lstm(model,X_test_final,Y_test_final):
  test_accuracy=0
  test_pred_tags=[]
  with torch.no_grad():
    for input_test, targets_test in zip(X_test_final,Y_test_final):
      input_test = torch.tensor(input_test, dtype=torch.long)
      targets_test = torch.tensor(targets_test, dtype=torch.long)

      model.zero_grad()
      tag_scores_test = model(input_test)

      _, indices_teat = torch.max(tag_scores_test, 1)
      test_pred_tags.append(indices_teat.tolist())
      test_accuracy = test_accuracy+calc_correct_lstm(targets_test,indices_teat)
    print(f"Test accuracy: {test_accuracy/len(mod_test_data)}")

    return test_pred_tags



def find_result_lstm(user_input_tensor,model,tag_vocab):
  mapped_tags=[]

  with torch.no_grad():
    model.zero_grad()
    tag_scores_test = model(user_input_tensor)
    _, ans1 = torch.max(tag_scores_test, 1)

    tag_mapping = {index: tag for tag, index in  tag_vocab.get_stoi().items()}

    mapped_tags = [tag_mapping[idx.item()] for idx in ans1]
  return mapped_tags

def run_model_on_user_input_lstm(model,tag_vocab,word_vocab,user_input):
  print("Your Sentense:", user_input)
  input= user_input.lower().split()
  input_encoded=sequence_to_idx_user(input,word_vocab)
  user_input_tensor = torch.tensor(input_encoded,dtype=torch.long)
  ans=find_result_lstm(user_input_tensor,model,tag_vocab)
  print("word","-->","tag")
  for word, tag in zip(input,ans):
     print(word,"-->",tag)

Y_valid,X_valid=replace_word_in_lists(Y_valid,X_valid,'SYM',)

if model_type=='-f':
   X_train=padding(X_train,"<s>","</s>",p,s)
   X_valid=padding(X_valid,"<s>","</s>",p,s)
   X_test=padding(X_test,"<s>","</s>",p,s)
   







X_train_encoded=sequence_to_idx(X_train,word_vocab)
Y_train_encoded=sequence_to_idx(Y_train,tag_vocab)



X_valid_encoded=sequence_to_idx(X_valid,word_vocab)
Y_valid_encoded=sequence_to_idx(Y_valid,tag_vocab)



X_test_encoded=sequence_to_idx(X_test,word_vocab)
Y_test_encoded=sequence_to_idx(Y_test,tag_vocab)

"""One-hot-encodeing"""


if model_type == '-f':
   X_train_one_hot_encoded=one_hot_encodeing_ffnn(X_train_encoded,len(word_vocab))
   y_train_one_hot_encoded=one_hot_encodeing_ffnn(Y_train_encoded,len(tag_vocab))
   X_valid_one_hot_encoded=one_hot_encodeing_ffnn(X_valid_encoded,len(word_vocab))
   y_valid_one_hot_encoded=one_hot_encodeing_ffnn(Y_valid_encoded,len(tag_vocab))
   X_test_one_hot_encoded=one_hot_encodeing_ffnn(X_test_encoded,len(word_vocab))
   y_test_one_hot_encoded=one_hot_encodeing_ffnn(Y_test_encoded,len(tag_vocab))
   X_mat, Y_mat = create_matrix_ffnn(X_train_one_hot_encoded, y_train_one_hot_encoded,p,s)
   X_mat_valid, Y_mat_valid = create_matrix_ffnn(X_valid_one_hot_encoded, y_valid_one_hot_encoded,p,s)
   X_mat_test, Y_mat_test = create_matrix_ffnn(X_test_one_hot_encoded, y_test_one_hot_encoded,p,s)
   X_train_input = np.array(X_mat)
   Y_train_input = np.array(Y_mat)
   X_valid_input = np.array(X_mat_valid)
   Y_valid_input = np.array(Y_mat_valid)
   X_test_input = np.array(X_mat_test)
   Y_test_input = np.array(Y_mat_test)
   embedding_dim = 601
   hidden_dim = 128
   output_dim = 13  # Number of POS tags
   batch_size = 32
   num_epochs = 50
   L_R=0.0001
   # Define the Feed Forward Neural Network model
   
   X_train_tensor = torch.tensor(X_train_input)
   y_train_tensor = torch.tensor(Y_train_input)
   X_valid_tensor = torch.tensor(X_valid_input)
   y_valid_tensor = torch.tensor(Y_valid_input)
   X_test_tensor = torch.tensor(X_test_input)
   y_test_tensor = torch.tensor(Y_test_input)
   train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
   validation_loader = DataLoader(TensorDataset(X_valid_tensor, y_valid_tensor), batch_size=32)
   test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=1)


   model = FFNN_POS_Tagger(embedding_dim, p, s, hidden_dim, output_dim)
   model.load_state_dict(torch.load("./trained_model/ffnn_model_best.pt"))
   model.eval()

  # for train model make above two line comment and uncomment below 3 line

  #  criterion = nn.CrossEntropyLoss()
  #  optimizer = optim.Adam(model.parameters(), lr=L_R)
  #  epoch_v,val_v,loss_v=train_model_ffnn(model,criterion,optimizer,num_epochs,train_loader,validation_loader)
  # draw graph
  # plot_graph(epoch_v, loss_v,val_v,[],[])

  #  pred_vector=test_model_ffnn(model,test_loader)
  #  pred_vector = torch.argmax(pred_vector, dim=1)
  #  y_test_tensor = torch.argmax(y_test_tensor, dim=1)
  # styled_table,conf_matrix=confusion_matrix_draw(y_test_tensor,pred_vector)
  #  print(conf_matrix)
   while(True):
    user_input=input("Enter Input if you want to quit press q:")
    if user_input == 'q':
      break
    run_model_on_user_input_ffnn(model,word_vocab,tag_vocab,user_input)

  #  run_model_on_user_input_ffnn(model,word_vocab,tag_vocab,user_input ="An apple a day keeps the doctor away")




if model_type =='-r':
  X_train_final = X_train_encoded
  Y_train_final = Y_train_encoded


  X_valid_final = X_valid_encoded
  Y_valid_final = Y_valid_encoded


  X_test_final = X_test_encoded
  Y_test_final = Y_test_encoded

  input_dim = 64
  output_dim = 13
  hidden_dim = 64
  DROPOUT=0.5
  lr=0.005
  num_epochs = 50
  Bidirectional=True
  Layer_number=2

  # Model

  model = LSTMTagger(input_dim, hidden_dim, len(word_vocab), len(tag_vocab), DROPOUT)

  # loss_function = nn.CrossEntropyLoss()
  # optimizer = optim.Adam(model.parameters(), lr)
  # for train model make above two line comment and uncomment below 2 line

  model.load_state_dict(torch.load("./trained_model/lstm_model1.pt"))
  model.eval()

  # train_accuracy,train_loss,valid_accuracy,valid_loss,epoch_v=train_model_lstm(model,loss_function,optimizer,num_epochs,X_train_final,Y_train_final,X_valid_final,Y_valid_final)

  # plot graphs Testing


  # plot_graphs(epoch_v,valid_loss,train_loss,valid_accuracy,train_accuracy)
  # test_pred_tags=test_model_lstm(model,X_test_final,Y_test_final)


  """Info on test data"""
  # result,conf_matrix=confusion_matrix_draw(flatten_list(y_test_tensor),flatten_list(pred_vector))
  # calculate_ans_print_class_accuracy(conf_matrix)
  while(True):
    user_input=input("Enter Input if you want to quit press q:")
    if user_input == 'q':
       break
    run_model_on_user_input_lstm(model,tag_vocab,word_vocab,user_input)
  # run_model_on_user_input_lstm(model,tag_vocab,word_vocab,user_input ="An apple a day keeps the doctor away")

