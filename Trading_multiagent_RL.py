import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pickle
import sys



class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(36,50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50,100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100,32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32,4)
#         self.fc5 = nn.Linear(32,1)
#         self.tanh1 = nn.Tanh()
#         self.fc6 = nn.Linear(32,1)
#         self.tanh2 = nn.Tanh()
        self.fc7 = nn.Linear(32,2)
#         self.sigmoid1 = nn.Sigmoid()
        self.fc8 = nn.Linear(32,2)
#         self.sigmoid2 = nn.Sigmoid()
        
    def forward(self,x):
        out = torch.FloatTensor(x).view(1,-1)
        out = self.relu1(self.fc1(out))
        out = self.relu2(self.fc2(out))
        out = self.relu3(self.fc3(out))
#         sell_rate = self.tanh1(self.fc5(out))
#         purchase_rate = self.tanh2(self.fc6(out))
        open_for_sell = self.fc7(out)
        open_for_purchase = self.fc8(out)
        out = self.fc4(out)
        return out,open_for_sell,open_for_purchase
    
    
    
class Entity:
    def __init__(self):
        self.food = 100 #Kg
        self.money = 100.0 #Rs
        self.life = 100.0 #years
        self.hunger = 0
        self.brain = Net()
        self.gamma = 0.9
        self.optimizer = optim.Adam(params=self.brain.parameters(),lr=1e-3)
        self.reward = 0
        self.baseline = 50
        self.life_decisions = []
        self.all_rews = []
        
        #memory of past 10 actions
        self.memory = np.zeros(30)
        self.mem_cntr = 0
        self.alive = True
        
        #logits collected
        self.decisions_logits = []
        self.open_for_sell_logits = []
        self.open_for_purchase_logits = []
        
        #rates
        self.sell_rate = 1
        self.purchase_rate = 1
        
        #business availability
        self.open_for_sell = False
        self.open_for_purchase = False
        
    def action(self,partner):
        partner_sr,partner_pr,partner_os,partner_op = partner.sell_rate,partner.purchase_rate,partner.open_for_sell,partner.open_for_purchase
#         state = np.append(np.array([self.food,self.money,self.life,self.hunger,partner_sr,partner_pr,int(partner_os),int(partner_op)]),self.memory)
        state = np.append(np.array([self.food,self.money,self.life,self.hunger,int(partner_os),int(partner_op)]),self.memory)
        decisions_array,os,op = self.brain(state)
#         self.open_for_sell = False if os.item() < 0.5 else True
#         self.open_for_purchase = False if op.item() < 0.5 else True
        
        
        
        m = Categorical(torch.softmax(os,1))
        sampled = m.sample()
        self.open_for_sell_logits.append(m.log_prob(sampled))
        self.open_for_sell = sampled.bool().item()

        m = Categorical(torch.softmax(op,1))
        sampled = m.sample()
        self.open_for_purchase_logits.append(m.log_prob(sampled))
        self.open_for_purchase = sampled.bool().item()

#         self.sell_rate = 1+sr.item()
#         self.purchase_rate = 1+pr.item()
        self.sell_rate = 1
        self.purchase_rate = 1
        m = Categorical(torch.softmax(decisions_array,1))
        sampled = m.sample()
        self.decisions_logits.append(m.log_prob(sampled))
        decision = sampled.item()
        mem_indx = self.mem_cntr
        self.memory = np.append(self.memory,[decision,self.open_for_sell,self.open_for_purchase])[3:]
        self.mem_cntr += 1
        
        actions = {0:self.eat,1:self.walk,2:self.sell,3:self.purchase}
        self.life_decisions.append(actions[decision].__name__)
#         print(actions[decision].__name__)
        if (decision == 2) | (decision == 3):
            actions[decision](partner)
        else:
            actions[decision]()
    
    def eat(self):
        if self.food > 0:
            self.food = max(self.food-1,0)
            self.hunger = max(self.hunger-3,0)
        else:
            self.hunger += 1
        self.life -= 1
        if self.life <= 0:
            self.alive = False
    
    def walk(self):
        self.hunger += 1
        self.life -= 1
        if (self.hunger >= 10) | (self.life <= 0):
            self.alive = False
    
    def sell(self,partner):
        partner_op = partner.open_for_purchase
        if partner_op:
            self.food -= 1
            self.money += 1 * self.sell_rate
            partner.food += 1
            partner.money -= 1 * self.sell_rate
            
        self.hunger += 1
        self.life -= 1
        if (self.hunger >= 10) | (self.life <= 0):
            self.alive = False
    
    def purchase(self,partner):
        partner_os = partner.open_for_sell
#         print('partner.open_for_sell',partner.open_for_sell)
        if partner_os:
            self.food += 1
            self.money -= 1 * self.purchase_rate
            partner.food -= 1
            partner.money += 1 * self.purchase_rate
            
        self.hunger += 1
        self.life -= 1
        if (self.hunger >= 10) | (self.life <= 0):
            self.alive = False
            
            
    def get_discounted_scaled_rewards(self):
        rewards = [self.reward * self.gamma**i for i in range(100)]
        rewards = list(reversed(rewards))
        rewards = torch.FloatTensor(rewards)
#         rewards = (rewards - rewards.min())/((rewards - rewards.min())).max()
        return rewards
            
    def learn(self):
        
#         print('health x wealth',self.money*self.mem_cntr)
        
#         self.reward = (self.money*self.mem_cntr)/(self.money+self.mem_cntr) - self.baseline
        self.reward = self.money - self.baseline
        discounted_rewards = self.get_discounted_scaled_rewards()
        self.all_rews.append(self.reward)
        
        decision_states = torch.stack(self.decisions_logits)
        os_states = torch.stack(self.open_for_sell_logits)
        op_states = torch.stack(self.open_for_purchase_logits)
        
        self.decisions_logits = []
        self.open_for_sell_logits = []
        self.open_for_purchase_logits = []
        self.life_decisions = []
        
        loss_decision = -(decision_states) * discounted_rewards
        loss_os = -(os_states) * discounted_rewards
        loss_op = -(op_states) * discounted_rewards
        
        loss = loss_decision.mean() + loss_os.mean() + loss_op.mean()
        
#         print('loss',loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.food = 100 #Kg
        self.money = 100.0 #Rs
        self.life = 100.0 #years
        self.hunger = 0
        
        self.memory = np.zeros(30)
        self.mem_cntr = 0
        self.alive = True
        
        #rates
        self.sell_rate = 1
        self.purchase_rate = 1
        
        #business availability
        self.open_for_sell = False
        self.open_for_purchase = False
        
        
        
people = """Liam
Noah
Oliver
William
Elijah
James
Benjamin
Lucas
Mason
Ethan
Alexander
Henry
Jacob
Michael
Daniel
Logan
Jackson
Sebastian
Jack
Aiden
Owen
Samuel
Matthew
Joseph
Levi
Mateo
David
John
Wyatt
Carter
Julian
Luke
Grayson
Isaac
Jayden
Theodore
Gabriel
Anthony
Dylan
Leo
Lincoln
Jaxon
Asher
Christopher
Josiah
Andrew
Thomas
Joshua
Ezra
Hudson
Charles
Caleb
Isaiah
Ryan
Nathan
Adrian
Christian
Maverick
Colton
Elias
Aaron
Eli
Landon
Jonathan
Nolan
Hunter
Cameron
Connor
Santiago
Jeremiah
Ezekiel
Angel
Roman
Easton
Miles
Robert
Jameson
Nicholas
Greyson
Cooper
Ian
Carson
Axel
Jaxson
Dominic
Leonardo
Luca
Austin
Jordan
Adam
Xavier
Jose
Jace
Everett
Declan
Evan
Kayden
Parker
Wesley
Kai"""

people = people.split('\n')

population = {i:Entity() for i in people}
np.random.shuffle(people)
pairs = list(zip(people[:50],people[50:]))

print('started')
sys.stdout.flush()
#action = interact
gens_rew = []
pop_rews = {i:[] for i in people}
for k in range(1000000):
    if k % 1000 == 0:
        print('**************',k,'****************')
        sys.stdout.flush()
        pickle.dump(gens_rew,open('RL/gens_rew.pkl','wb'))
        pickle.dump(pop_rews,open('RL/pop_rews.pkl','wb'))
        pickle.dump(population,open('RL/population.pkl','wb'))
    pop_rew = []
    for i,j in pairs:
        entity_1 = population[i]
        entity_2 = population[j]

#         print(i,{l:m for l,m in vars(entity_1).items() if (l != 'brain') & (l != 'memory')})
#         print(j,{l:m for l,m in vars(entity_2).items() if (l != 'brain') & (l != 'memory')})
        
        entity_1.action(entity_2)
        entity_2.action(entity_1)
        
#         print(i,{l:m for l,m in vars(entity_1).items() if (l != 'brain') & (l != 'memory')})
#         print(j,{l:m for l,m in vars(entity_2).items() if (l != 'brain') & (l != 'memory')})
        
#         print('ln',len({i:j for i,j in population.items() if j.alive == False}))

        if entity_1.alive == False:
            entity_1.learn()
            pop_rews[i].append(entity_1.reward)
            pop_rew.append(entity_1.reward)
        if entity_2.alive == False:
            entity_2.learn()
            pop_rews[j].append(entity_2.reward)
            pop_rew.append(entity_2.reward)       
#     print(np.mean(pop_rew))
    gens_rew.append(np.mean(pop_rew))
    np.random.shuffle(people)
    pairs = list(zip(people[:50],people[50:]))
    
#     print('ln_pop',len(population))

