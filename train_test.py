import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model.widar3_torch import Widar3_torch
import read_bvp



ALL_MOTION = [1, 2, 3, 4, 5, 6]
N_MOTION = len(ALL_MOTION)
batch_size = 32

full_dataset = read_bvp.BVPDataSet(data_dir="/home/ubuntu/new_bvp/testdata/user1", motion_sel=ALL_MOTION)
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


# def get_train_device():
#     device = torch.device("cpu")
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")

#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#     return device


device = "cuda:3"
EPOCH = 1000
TIME_STEPS = full_dataset.get_T_max()


# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
model = Widar3_torch(6)
model.to(device)
print(model)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 300, 800], gamma=0.1)

for epoch in range(EPOCH):
    #### train ####
    model.train()
    train_loss_sum = 0.0
    train_correct_sum = 0
    # train_sample_sum = 0
    for idx, data in enumerate(train_loader):
        img, label = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(img)
        train_batch_loss = criterion(output, label)
        train_batch_loss.backward()
        optimizer.step()
        train_actual_label = torch.max(label.data, 1)[1]
        train_predicted_label = torch.max(output.data, 1)[1]
        train_correct_sum += (train_actual_label == train_predicted_label).sum()
        train_loss_sum += train_batch_loss.item()
    train_epoch_loss = train_loss_sum / (idx+1)
    train_epoch_acc = (train_correct_sum.item())/len(train_dataset)*100

    print(f"Epoch [{epoch}] lr: {optimizer.state_dict()['param_groups'][0]['lr']}", end=" ")
    print('Train epoch loss:', train_epoch_loss, end=" ")
    print(f'Train epoch acc: {train_epoch_acc}%', end=" ")
    scheduler.step()

    #### validation ####
    model.eval()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            test_loss += criterion(pred, label).item()
            correct += (pred.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")

