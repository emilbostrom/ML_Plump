import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden1_size,hidden2_size,hidden3_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden1_size)
        self.linear2 = nn.Linear(hidden1_size,hidden2_size)
        self.linear3 = nn.Linear(hidden2_size,hidden3_size)
        self.linear4 = nn.Linear(hidden3_size,output_size)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        return x

    def save(self,file_name='model2.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self,state,next_state,cardGuess,reward,done):
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        cardGuess = torch.tensor(cardGuess,dtype=torch.float)
        reward = torch.tensor(reward,dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            cardGuess = torch.unsqueeze(cardGuess,0)
            reward = torch.unsqueeze(reward,0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()

        for idx in range(len(done)):
            Q_new  = reward[idx]
        
            if len(done) > 1:
                #done_idx= done[idx][0][0]
                done_idx= done[idx]
            else:
                done_idx = done[idx][0]

            if not done_idx:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            if len(done) > 1:
                #target[idx][0,int(cardGuess[idx,0])] = Q_new
                target[idx,int(cardGuess[idx])] = Q_new
            else:
                target[idx,int(cardGuess[idx,0])] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -- Only do this if not done
        # pred.clone()
        # pred[argmax(cardGuess)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimizer.step()