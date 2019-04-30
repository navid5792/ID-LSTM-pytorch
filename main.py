import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.LSTM import LSTMClassifier
import torch.nn as nn
from copy import deepcopy
import random
from tqdm import tqdm

learning_rate = 2e-5
batch_size = 5
global_batch_size = 5
output_size = 2
hidden_size = 300
embedding_length = 300
samplecnt = 5
epsilon = 0.05
maxlength = 200
alpha = 0.1
tau = 0.1
delay_critic = True

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset(batch_size=batch_size)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def Sampling_RL(actor, critic, inputs, vector, length, epsilon, Random = True):
    current_lower_state = torch.zeros(1,2*hidden_size).cuda()
    actions = []
    states = []
    for pos in range(length):
        predicted = actor.get_target_output(current_lower_state, vector[0][pos], scope = "target")
        states.append([current_lower_state, vector[0][pos]])
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < float(predicted[0].item()) else 1)
            else:
                action = (1 if random.random() < float(predicted[0].item()) else 0)
        else:
            action = np.argmax(predicted).item()
        actions.append(action)
        if action == 1:
            out_d, current_lower_state = critic.forward_lstm(current_lower_state, inputs[0][pos], scope = "target")
    Rinput = []
    for (i, a) in enumerate(actions):
        if a == 1:
            Rinput.append(int(inputs[0][i].item())) ####
    Rlength = len(Rinput)
    #print("problem")
    if Rlength == 0:
        actions[length-2] = 1
        Rinput.append(inputs[0][length-2])
        Rlength = 1
    Rinput += [1] * (maxlength - Rlength)

    Rinput = torch.tensor(Rinput).view(1,-1).cuda()
    
    return actions, states, Rinput, Rlength


class policyNet(nn.Module):
    def __init__(self):
        super(policyNet, self).__init__()
        self.hidden = hidden_size
        self.W1 = nn.Parameter(torch.cuda.FloatTensor(2*self.hidden, 1).uniform_(-0.5, 0.5)) 
        self.W2 = nn.Parameter(torch.cuda.FloatTensor(embedding_length, 1).uniform_(-0.5, 0.5)) 
        self.b = nn.Parameter(torch.cuda.FloatTensor(1, 1).uniform_(-0.5, 0.5))

    def forward(self, h, x):
        h_ = torch.matmul(h.view(1,-1), self.W1) # 1x1
        x_ = torch.matmul(x.view(1,-1), self.W2) # 1x1
        scaled_out = torch.sigmoid(h_ +  x_ + self.b) # 1x1
        scaled_out = torch.clamp(scaled_out, min=1e-5, max=1 - 1e-5)
        scaled_out = torch.cat([1.0 - scaled_out, scaled_out],0)
        return scaled_out



class critic(nn.Module):
    def __init__(self):
        super(critic, self).__init__()
        self.target_pred = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
        self.active_pred = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)


    def forward(self, x, scope):
        if scope == "target":
            out = self.target_pred(x)
        if scope == "active":
            out = self.active_pred(x)
        return out

    def assign_target_network(self):
        params = []
        for name, x in self.active_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

    def update_target_network(self):
        params = []
        for name, x in self.active_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_pred.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name, x in self.target_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_pred.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

    def assign_active_network_gradients(self):
        params = []
        for name, x in self.target_pred.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_pred.named_parameters():
            x.grad = deepcopy(params[i].grad)
            i+=1
        for name, x in self.target_pred.named_parameters():
            x.grad = None

    def forward_lstm(self, hc, x, scope):
        if scope == "target":
            out, state = self.target_pred.getNextHiddenState(hc, x)
        if scope == "active":
            out, state = self.active_pred.getNextHiddenState(hc, x)
        return out, state

    def wordvector_find(self, x):
        return self.target_pred.wordvector_find(x)


class actor(nn.Module):
    def __init__(self):
        super(actor, self).__init__()
        self.target_policy = policyNet()
        self.active_policy = policyNet()
        
    def get_target_logOutput(self, h, x):
        out = self.target_policy(h, x)
        logOut = torch.log(out)
        return logOut

    def get_target_output(self, h, x, scope):
        if scope == "target":
            out = self.target_policy(h, x)
        if scope == "active":
            out = self.active_policy(h, x)
        return out

    def get_gradient(self, h, x, reward, scope):
        if scope == "target":
            out = self.target_policy(h, x)
            logout = torch.log(out).view(-1)
            index = reward.index(0)
            index = (index + 1) % 2
            #print(out, reward, index, logout[index].view(-1), logout)
            #print(logout[index].view(-1))
            grad = torch.autograd.grad(logout[index].view(-1), self.target_policy.parameters()) # torch.cuda.FloatTensor(reward[index])
            #print(grad[0].size(), grad[1].size(), grad[2].size())
            #print(grad[0], grad[1], grad[2])
            grad[0].data = grad[0].data * reward[index]
            grad[1].data = grad[1].data * reward[index]
            grad[2].data = grad[2].data * reward[index]
            #print(grad[0], grad[1], grad[2])
            return grad
        if scope == "active":
            out = self.active_policy(h, x)
        return out
    def assign_active_network_gradients(self, grad1, grad2, grad3):
        params = [grad1, grad2, grad3]    
        i=0
        for name, x in self.active_policy.named_parameters():
            x.grad = deepcopy(params[i])
            i+=1

    def update_target_network(self):
        params = []
        for name, x in self.active_policy.named_parameters():
            params.append(x)
        i=0
        for name, x in self.target_policy.named_parameters():
            x.data = deepcopy(params[i].data * (tau) + x.data * (1-tau))
            i+=1

    def assign_active_network(self):
        params = []
        for name, x in self.target_policy.named_parameters():
            params.append(x)
        i=0
        for name, x in self.active_policy.named_parameters():
            x.data = deepcopy(params[i].data)
            i+=1

def train_model(criticModel, actorModel, train_iter, epoch, RL_train = True, LSTM_train = True):
    total_epoch_loss = 0
    total_epoch_acc = 0
    criticModel.cuda()
    actorModel.cuda()
    critic_target_optimizer = torch.optim.Adam(criticModel.target_pred.parameters())
    critic_active_optimizer = torch.optim.Adam(criticModel.active_pred.parameters())

    actor_target_optimizer = torch.optim.Adam(actorModel.target_policy.parameters())
    actor_active_optimizer = torch.optim.Adam(actorModel.active_policy.parameters())
    steps = 0
    for idx, batch in enumerate(train_iter):
        if idx % 100 == 0:
            print(idx , "/", len(train_iter))
        totloss = 0.
        text = batch.text[0]
        target = batch.label
        lengths = batch.text[1]
        target = torch.autograd.Variable(target).long()
        pred = torch.zeros(batch_size, 2).cuda()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not batch_size):# One of the batch returned by BucketIterator has length different than 32.
            continue
        #if steps % 50 == 0:
            #print(actorModel.target_policy.b.data, actorModel.active_policy.b.data)
        criticModel.assign_active_network()
        actorModel.assign_active_network()
        #if steps % 50 == 0:
            #print(actorModel.target_policy.b.data, actorModel.active_policy.b.data)
        #print(actorModel.target_policy.W1, actorModel.active_policy.W1, "\n\n", criticModel.target_pred.label.bias, criticModel.active_pred.label.bias)
        avgloss = 0
        aveloss = 0.
        for i in range(batch_size):
            x = text[i].view(1,-1)
            y = target[i].view(1)
            length = int(lengths[i])
            if RL_train:
                #print("RL True")
                criticModel.train(False)
                actorModel.train()
                actionlist, statelist, losslist = [], [], []
                aveLoss = 0.
                for i in range(samplecnt):
                    actions, states, Rinput, Rlength = Sampling_RL(actorModel, criticModel, x, criticModel.wordvector_find(x), length, epsilon, Random=True)
                    '''
                    if (steps) % 50 == 0:
                        criticModel.eval()
                        actorModel.eval()
                        act, _, _, _ = Sampling_RL(actorModel, criticModel, x, criticModel.wordvector_find(x), length, epsilon, Random=False)
                        print(act, "\n\n")
                        criticModel.train()
                        actorModel.train()
                    '''
                    actionlist.append(actions)
                    statelist.append(states)
                    out = criticModel(Rinput, scope = "target")
                    loss_ = loss_fn(out, y)
                    loss_ += (float(Rlength) / length) **2 *0.15
                    aveloss += loss_
                    losslist.append(loss_)
                '''
                if (steps) % 50 == 0:
                    print("-------------------------------------------")
                '''
                aveloss /= samplecnt
                totloss += aveloss
                grad1 = None
                grad2 = None
                grad3 = None
                flag = 0 
                if LSTM_train:
                    #print("RL and LSTM True")
                    criticModel.train()
                    actorModel.train()  
                    critic_active_optimizer.zero_grad()
                    critic_target_optimizer.zero_grad()
                    prediction = criticModel(Rinput, scope = "target")
                    pred[i] = prediction
                    loss = loss_fn(prediction, y)
                    loss.backward()
                    #print(criticModel.active_pred.label.bias.grad, criticModel.target_pred.label.bias.grad)
                    #print(criticModel.active_pred.label.bias, criticModel.target_pred.label.bias)
                    criticModel.assign_active_network_gradients()
                    #print(criticModel.active_pred.label.bias.grad, criticModel.target_pred.label.bias.grad)
                    critic_active_optimizer.step()
                    #print(criticModel.active_pred.label.bias, criticModel.target_pred.label.bias)
                for i in range(samplecnt):
                    for pos in range(len(actionlist[i])):
                        rr = [0, 0]
                        rr[actionlist[i][pos]] = ((losslist[i] - aveloss) * alpha).cpu().item()
                        g = actorModel.get_gradient(statelist[i][pos][0], statelist[i][pos][1], rr, scope = "target")
                        if flag == 0:
                            grad1 = g[0]
                            grad2 = g[1]
                            grad3 = g[2]
                            flag = 1
                        else:
                            grad1 += g[0]
                            grad2 += g[1]
                            grad3 += g[2]
                        #print("++", grad3)
                #print("\n\n before: active: ", actorModel.active_policy.b, "target: ", actorModel.target_policy.b, "gradient to be applied: ", grad3)
                actor_target_optimizer.zero_grad()
                actor_active_optimizer.zero_grad()
                #print("previous grad: ", actorModel.active_policy.b.grad)
                actorModel.assign_active_network_gradients(grad1, grad2, grad3)
                actor_active_optimizer.step()
                #print("after: active: ", actorModel.active_policy.b, "target: ", actorModel.target_policy.b)
            else: 
                #print("RL False LSTM True")
                criticModel.train()
                actorModel.train(False)  
                critic_active_optimizer.zero_grad()
                critic_target_optimizer.zero_grad()
                prediction = criticModel(x, scope = "target")
                pred[i] = prediction
                loss = loss_fn(prediction, y)
                avgloss += loss.item()
                loss.backward()
                criticModel.assign_active_network_gradients()
                critic_active_optimizer.step()
        
        if RL_train:
            #print("Again RL True")
            criticModel.train(False)
            actorModel.train()
            #print(actorModel.target_policy.b.data, actorModel.active_policy.b.data)
            actorModel.update_target_network()
            #print(actorModel.target_policy.b.data, actorModel.active_policy.b.data)
            if LSTM_train:
                #print("Again RL AND LSTM True")
                criticModel.train()
                actorModel.train() 
                #print(criticModel.active_pred.label.bias, criticModel.target_pred.label.bias)
                criticModel.update_target_network()
                #print(criticModel.active_pred.label.bias, criticModel.target_pred.label.bias)
                
        else:
            #print("Again RL False and LSTM True")
            criticModel.train()
            actorModel.train(False)  
            criticModel.assign_target_network()
        avgloss /= batch_size
        num_corrects = (torch.max(pred, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        steps += 1
        
        #if steps % 50 == 0:
            #print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {avgloss:.4f}, Training Accuracy: {acc.item(): .2f}%')  
            #print(actorModel.target_policy.b.data, actorModel.active_policy.b.data) 
        
        total_epoch_loss += avgloss
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


def train_model_without_delay(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.target_pred.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text, scope = "target")
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not batch_size):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text, scope = "target")
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)

def eval_model_RL(criticModel, actorModel, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    criticModel.eval()
    actorModel.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            if idx % 100 == 0:
                print(idx, "/", len(val_iter))
            text = batch.text[0]
            if (text.size()[0] is not batch_size):
                continue
            target = batch.label
            lengths = batch.text[1]
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            batch_loss = 0
            pred = torch.zeros(batch_size, 2).cuda()
            for i in range(batch_size):
                x = text[i].view(1,-1)
                y = target[i].view(1)
                length = int(lengths[i])

                actions, states, Rinput, Rlenth = Sampling_RL(actorModel, criticModel, x, criticModel.wordvector_find(x), length, epsilon, Random=False)
                #print(x, Rinput, length, Rlenth)
                #if (i % 50) == 0:
                    #print(actions)
                prediction = criticModel(Rinput, scope = "target")
                loss = loss_fn(prediction, y)
                batch_loss += loss
                pred[i] = prediction
            num_corrects = (torch.max(pred, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += batch_loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)

criticModel = critic()
actorModel = actor()

actorModel.cuda()
criticModel.cuda()

loss_fn = F.cross_entropy
best_val_acc = 0.

criticModel.load_state_dict(torch.load('savedModels/critic_with_delay.pt'))
_, best_val_acc = eval_model(criticModel, valid_iter)
print(best_val_acc)
_, best_val_acc = eval_model(criticModel, train_iter)
print(best_val_acc)

if delay_critic:
    for epoch in range(0):
        print("Pre-training Critic...")
        train_loss, train_acc = train_model(criticModel, actorModel, train_iter, epoch, RL_train = False)
        val_loss, val_acc = eval_model(criticModel, valid_iter)
        if val_acc > best_val_acc:
            torch.save(criticModel.state_dict(), 'savedModels/critic_with_delay.pt')
            best_val_acc = val_acc
            print("saved Model with acc: ", val_acc)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
else:
    for epoch in range(0):
        train_loss, train_acc = train_model_without_delay(criticModel, train_iter, epoch)
        val_loss, val_acc = eval_model(criticModel, valid_iter)
        if val_acc > best_val_acc:
            torch.save(criticModel.state_dict(), 'savedModels/critic_without_delay.pt')
            best_val_acc = val_acc
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

#val_loss, val_acc = eval_model(criticModel, valid_iter)
#test_loss, test_acc = eval_model(criticModel, test_iter)
#train_loss, train_acc = eval_model(criticModel, train_iter)
epoch = 0
#print("LSTM Pretraining Done: ")
#print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:3f}, Train Acc: {train_acc:.2f}, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}, Test. Loss: {test_loss:3f}, Test. Acc: {test_acc:.2f}%')

'''
val_loss, val_acc = eval_model_RL(criticModel, actorModel,  valid_iter)
print(val_loss, val_acc)
asaas
'''

'''
val_loss, val_acc = eval_model_RL(criticModel, actorModel, valid_iter)
print(f'Epoch: {epoch+1:02}, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
asas
'''
'''
actorModel.load_state_dict(torch.load('savedModels/actor_with_delay.pt'))
print("Model loaded after epoch 10")
print("Starting Reinforcement....")
_, best_val_acc2 = eval_model_RL(criticModel, actorModel, valid_iter)
print(best_val_acc2)
_, best_val_acc2 = eval_model_RL(criticModel, actorModel, train_iter)
print(best_val_acc2)
'''

best_val_acc1 = 810.5
for epoch in range(0):
    train_loss, train_acc = train_model(criticModel, actorModel, train_iter, epoch, LSTM_train = False)
    val_loss, val_acc = eval_model_RL(criticModel, actorModel,  valid_iter)
    if val_acc > best_val_acc1:
        torch.save(actorModel.state_dict(), 'savedModels/actor_with_delay.pt')
        best_val_acc1 = val_acc
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
print("Reinforcement Done!!!!")
'''
actorModel.load_state_dict(torch.load('savedModels/actor_with_delay.pt'))
print("Model Loaded..")

val_loss, val_acc = eval_model_RL(criticModel, actorModel,  valid_iter)
print(val_acc)
'''
criticModel.load_state_dict(torch.load('savedModels/critic_with_delay_joint.pt'))
actorModel.load_state_dict(torch.load('savedModels/actor_with_delay_joint.pt')) 
_, best_val_acc2 = eval_model_RL(criticModel, actorModel, valid_iter)
print(best_val_acc2)
_, best_val_acc2 = eval_model_RL(criticModel, actorModel, train_iter)
print(best_val_acc2)
asasa
for epoch in range(0):
    train_loss, train_acc = train_model(criticModel, actorModel, train_iter, epoch)
    val_loss, val_acc = eval_model_RL(criticModel, actorModel, valid_iter)
    print(val_acc)
    if val_acc > best_val_acc2:
        torch.save(actorModel.state_dict(), 'savedModels/actor_with_delay_joint.pt')
        torch.save(criticModel.state_dict(), 'savedModels/critic_with_delay_joint.pt')
        best_val_acc2 = val_acc  
        print("----Mdoel Saved-----") 
    

criticModel.load_state_dict(torch.load('savedModels/critic_with_delay_joint.pt'))
actorModel.load_state_dict(torch.load('savedModels/actor_with_delay_joint.pt')) 
test_loss, test_acc = eval_model_RL(criticModel, actorModel, valid_iter)
print(test_acc)
test_loss, test_acc = eval_model_RL(criticModel, actorModel, test_iter)
print(test_acc)

'''
#Let us now predict the sentiment on a single sentence just for the testing purpose

test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

test_sen2 = TEXT.preprocess(test_sen2)
test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
test_tensor = Variable(test_sen, volatile=True)
test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print ("Sentiment: Positive")
else:
    print ("Sentiment: Negative")
'''