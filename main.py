from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import device
from dataset import TextDataset
from model_utils import unfreeze_layers, compute_detailed_scores_pytorch
from training import train_model

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('moussaKam/barthez-orangesum-abstract')
tokenizer = AutoTokenizer.from_pretrained('moussaKam/barthez-orangesum-abstract')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_to_train = pd.read_csv("/content/drive/My Drive/data_complete2.csv")
data_to_train = data_to_train.iloc[:, [0, 1]]

for param in model.parameters():
    param.requires_grad = False

unfreeze_layers(model, "encoder", [-1])
unfreeze_layers(model, 'decoder', [-1])

df_train, df_val = train_test_split(data_to_train, test_size=0.15)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# Create the DataLoaders
epoch = 4
train_dataset = TextDataset(df_train, tokenizer, max_length=1024)
train_loader = DataLoader(train_dataset, batch_size=2)
val_dataset = TextDataset(df_val, tokenizer, max_length=1024)
val_loader = DataLoader(val_dataset, batch_size=2)

# Train model
train_model(model, train_loader, val_loader, device, epoch)
