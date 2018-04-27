# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:06:29 2018

@author: Kumaraguru
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import random
#from numpy import linalg as LA

#building the simulator
class gym(object):
    def __init__(self,sqgrid):
        self.val1 =180
        self.val2 =213
        self.val3=100
        self.val4 =180
        self.val5 =213
        self.val6=100
        self.val7=200
        self.val8=0
        self.val9=0
        self.val10=0
        self.val11=0
        self.val12=255
        self.sqgrid = sqgrid
        self.side = 2*sqgrid+1
        self.size = self.side*self.side
        self.frame = 0
        self.baseval1=100
        self.baseval2=-100
        self.episode=0
        self.act=sqgrid
        self.weight = 10*np.array(([-1,-1,-1],[-1,10,-1],[-1,-1,-1]),dtype=np.int32)
        return None
    def make(self,name):
        self.environment = name
        return self.environment
    def reset(self):
        self.state = np.zeros((self.side,self.side), dtype=np.int32)
        self.indexpos1 = [ num for num in range(1,self.side-1) if (num+1) % 2]
        self.indexpos2 = [ num for num in range(1,self.side) if num % 2]
        if np.random.randint(0,100)>50:
            indexpos = self.indexpos1
        else:
            indexpos = self.indexpos2
        self.a = np.random.choice(indexpos)
        self.b = np.random.choice(indexpos)
        if np.random.randint(0,100)>50:
            indexpos = self.indexpos1
        else:
            indexpos = self.indexpos2
        self.c = np.random.choice(indexpos)
        self.d = np.random.choice(indexpos)
        self.state[self.a,self.b]=self.baseval1
        self.state[self.c,self.d]=self.baseval2
        return self.state
    
    def render(self,matrix):
        plt.close()
        self.ax = plt.axes()
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(12)) 
        self.ax.imshow(matrix)
        #plt.pause(0)
        #plt.savefig('Episode'+str(self.episode)+'_frame'+str(self.frame)+'.png',dpi=200)
    
    def action_random(self):
        self.on= np.random.randint(-1,2,size=(self.act,self.act))
        self.akron = np.kron(self.on,([1,1],[1,1]))
        self.atile = np.tile(([1,0],[0,0]),(self.sqgrid,self.sqgrid))
        self.amul = np.multiply(self.akron,self.atile)
        self.apad = np.pad(self.amul,(1,0),'constant',constant_values=0)
        return self.apad
    
    def stepenv(self,action):
        self.actconv= ndimage.convolve(action,self.weight,cval=0)
        self.actconv=self.actconv+np.random.randint(-2,3,size=(self.side,self.side))
        if (self.a in self.indexpos1):
            print('p1: ',end="")
            print(self.a,self.b)
        else:
            print('p2: ',end="")
            print(self.a,self.b)

        self.iter1=self.a-1
        self.iter2=self.b-1
        self.matrixaction=[]
        for i in range(0,3):
            self.iter2 = self.b-1
            for j in range(0,3):
                try:
                    self.matrixaction.append(self.actconv[self.iter1,self.iter2])
                except IndexError: 
                    self.matrixaction.append(-1000)
                self.iter2+=1
            self.iter1+=1
        self.actiontotake = 1+np.argmax(self.matrixaction)
        return self.actiontotake
    
    def step(self, actionnum):
        self.actiontotake=actionnum
        self.state[self.a,self.b]=0
        try:
            if(self.actiontotake==1):
                self.state[self.a-1,self.b-1]=self.baseval1
                self.a = self.a-1
                self.b = self.b-1
            elif(self.actiontotake==2):
                self.state[self.a-1,self.b]=self.baseval1
                self.a = self.a-1
                self.b = self.b
            elif(self.actiontotake==3):
                self.state[self.a-1,self.b+1]=self.baseval1
                self.a = self.a-1
                self.b = self.b+1
            elif(self.actiontotake==4):
                self.state[self.a,self.b-1]=self.baseval1
                self.a = self.a
                self.b = self.b-1
            elif(self.actiontotake==5):
                self.state[self.a,self.b]=self.baseval1
                self.a = self.a
                self.b = self.b
            elif(self.actiontotake==6):
                self.state[self.a,self.b+1]=self.baseval1
                self.a = self.a
                self.b = self.b+1
            elif(self.actiontotake==7):
                self.state[self.a+1,self.b-1]=self.baseval1
                self.a = self.a+1
                self.b = self.b-1
            elif(self.actiontotake==8):
                self.state[self.a+1,self.b]=self.baseval1
                self.a = self.a+1
                self.b = self.b
            else:
                self.state[self.a+1,self.b+1]=self.baseval1 
                self.a = self.a+1
                self.b = self.b+1
        except IndexError:
                self.state[self.a,self.b]=self.baseval1
                self.a = self.a
                self.b = self.b
        self.render2D(self.state)
        if (self.a==self.c and self.b==self.d):
            self.reward=1000
            self.done = 1
        else:
            self.reward=-1
            self.done =0
        self.D3state = self.get3D()
        return self.D3state, self.reward, self.done

    def render2D(self,matrix):
        plt.close()
        self.ax = plt.axes()
        #self.ax.grid(color='y', linestyle='-', linewidth=1)
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(12)) 
        self.ax.imshow(matrix)
        #plt.pause(0)
        #plt.savefig('Episode'+str(self.episode)+'_frame'+str(self.frame)+'2D.png',dpi=200)
    
    def get3D(self):
        self.D3size=112
        self.D3state = np.zeros((self.D3size, self.D3size,3), dtype=np.uint8)
        j=5
        k=5
        for ii in range(0,int(self.D3size/10)):
            for i in range(0,int(self.D3size/10)):
                self.D3state[j,k,0]=self.val1
                self.D3state[j+1,k+1,0]=self.val1
                self.D3state[j+1,k,0]=self.val1
                self.D3state[j,k+1,0]=self.val1
                
                self.D3state[j,k,1]=self.val2
                self.D3state[j+1,k+1,1]=self.val2
                self.D3state[j+1,k,1]=self.val2
                self.D3state[j,k+1,1]=self.val2
                
                self.D3state[j,k,2]=self.val3
                self.D3state[j+1,k+1,2]=self.val3
                self.D3state[j+1,k,2]=self.val3
                self.D3state[j,k+1,2]=self.val3
                j=j+10
            k=k+10
            j=5
        j=0
        k=0
        for ii in range(0,1+int(self.D3size/10)):
            for i in range(0,1+int(self.D3size/10)):
                self.D3state[j,k,0]=self.val4
                self.D3state[j+1,k+1,0]=self.val4
                self.D3state[j+1,k,0]=self.val4
                self.D3state[j,k+1,0]=self.val4
                
                self.D3state[j,k,1]=self.val5
                self.D3state[j+1,k+1,1]=self.val5
                self.D3state[j+1,k,1]=self.val5
                self.D3state[j,k+1,1]=self.val5
                
                self.D3state[j,k,2]=self.val6
                self.D3state[j+1,k+1,2]=self.val6
                self.D3state[j+1,k,2]=self.val6
                self.D3state[j,k+1,2]=self.val6
                j=j+10
            k=k+10
            j=0
        
        self.starta = self.a*5
        self.startb = self.b*5
        self.startc = self.c*5
        self.startd = self.d*5
        #print('p4: ',end="")
        #print(self.episode, self.frame,self.starta,self.startb)
        try:
            self.D3state[self.starta,self.startb,0]=self.val7
            self.D3state[self.startc,self.startd,0]=self.val10
            self.D3state[self.starta+1,self.startb,0]=self.val7       
            self.D3state[self.startc+1,self.startd,0]=self.val10
            self.D3state[self.starta,self.startb+1,0]=self.val7     
            self.D3state[self.startc,self.startd+1,0]=self.val10
            self.D3state[self.starta+1,self.startb+1,0]=self.val7       
            self.D3state[self.startc+1,self.startd+1,0]=self.val10   
            self.D3state[self.starta,self.startb,1]=self.val8
            self.D3state[self.startc,self.startd,1]=self.val11
            self.D3state[self.starta+1,self.startb,1]=self.val8       
            self.D3state[self.startc+1,self.startd,1]=self.val11
            self.D3state[self.starta,self.startb+1,1]=self.val8     
            self.D3state[self.startc,self.startd+1,1]=self.val11
            self.D3state[self.starta+1,self.startb+1,1]=self.val8       
            self.D3state[self.startc+1,self.startd+1,1]=self.val11
            self.D3state[self.starta,self.startb,2]=self.val9
            self.D3state[self.startc,self.startd,2]=self.val12
            self.D3state[self.starta+1,self.startb,2]=self.val9       
            self.D3state[self.startc+1,self.startd,2]=self.val12
            self.D3state[self.starta,self.startb+1,2]=self.val9     
            self.D3state[self.startc,self.startd+1,2]=self.val12
            self.D3state[self.starta+1,self.startb+1,2]=self.val9       
            self.D3state[self.startc+1,self.startd+1,2]=self.val12
        except IndexError:
            pass
        self.render(self.D3state)
        return self.D3state   
        
plt.close()
state_seq =[]
env = gym(11)    
game = env.make('micro')   

observation = env.reset()        
image_sequence=list()
memory = list()
img=observation    

#initialize the weights
W_conv1 = tf.Variable(tf.truncated_normal([8,8,4,16], stddev = 0.01), name="W_conv1")
b_conv1 = tf.Variable(tf.constant(0.01, shape = [16]), name="b_conv1")
W_conv2 = tf.Variable(tf.truncated_normal([4,4,16,32], stddev = 0.01), name="W_conv2")
b_conv2 = tf.Variable(tf.constant(0.01, shape = [32]),name="b_conv2")
W_fc1 = tf.Variable(tf.truncated_normal([2592, 256], stddev = 0.01),name="W_fc1")
b_fc1 = tf.Variable(tf.constant(0.01, shape = [256]),name="b_fc1")
W_fc2 = tf.Variable(tf.truncated_normal([256, 9], stddev = 0.01),name="W_fc2")
b_fc2 = tf.Variable(tf.constant(0.01, shape =[9]),name="b_fc2")

observation_image = tf.placeholder(tf.float32, shape=(112,112,3))
image_input = tf.placeholder(tf.float32, shape=(None,4,84,84,1))

#preprocess images
def process_image(img):
    gray_scaled_image = tf.image.rgb_to_grayscale(img)
    resized_image = tf.image.resize_images(gray_scaled_image, [110,84])
    cropped_image = tf.image.resize_image_with_crop_or_pad(resized_image,84,84) 
    return cropped_image
    
#convert action into one hot vector
def get_action_matrix(val):
    actions = np.zeros(9)
    actions[val]=1
    return actions

#perform forward propagation
def forward_pass(img):
    img = tf.reshape(img,shape=[-1,84,84,4])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(img, W_conv1, [1, 4, 4, 1], padding = "VALID") + b_conv1)
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, [1, 2, 2, 1], "VALID") + b_conv2)
    h_conv2_flat = tf.reshape(h_conv2,[-1, 2592])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    out = tf.matmul(h_fc1,W_fc2) + b_fc2
    return out

processed_image= process_image(observation_image)
q = forward_pass(image_input)
action_index = tf.argmax(q,1)


# Training
actionInput = tf.placeholder("float", [None, 9])
yInput = tf.placeholder("float", [None])
Q_Action = tf.reduce_sum(tf.multiply(q, actionInput), reduction_indices = 1)
cost = tf.reduce_mean(tf.square(yInput - Q_Action))
trainStep = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(cost)
train_int=2000
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#run tensorflow sessions
with tf.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    sess.run(init)
    epsilon=0.9
    global_step = 0
    for episode in range(0,200):
       #print("p3: Initializing new episode: ",end="")
       observation = env.reset()
       reward_sum = 0
       print(env.a,env.b,env.c,env.d)
       env.episode+=1
       env.frame=0    
       observation = env.get3D()

       while True:
            #print(str(env.episode)+' '+str(env.frame)+' ', str(reward_sum))
            p_image = sess.run(processed_image, feed_dict={observation_image:observation})
            image_sequence.append(p_image)
                        
            if len(image_sequence)<=4:
                actiontotake=np.random.randint(0,10)
                next_observation, reward, done = env.step(actiontotake)
                #print('I am executed')
            else:
                image_sequence.pop(0)
                current_state = np.stack([image_sequence[0],image_sequence[1],image_sequence[2],image_sequence[3]])
                #print('I am also executed')
                epsilon=max(0.2, epsilon*0.95)
                if np.random.rand(1)<epsilon or episode<10:
                    action = np.random.randint(0,10)
                    #print('random action')
                else:
                    qval = sess.run(q, feed_dict={image_input:np.array(current_state).reshape(1,4,84,84,1)})
                    action = sess.run(action_index, feed_dict={q:qval})
                    #print("Q-action")   
                next_observation, reward, done = env.step(action-1)
                p_image = sess.run(processed_image, feed_dict={observation_image:observation})
                next_state = np.stack([image_sequence[1],image_sequence[2],image_sequence[3],p_image])
                action_state = get_action_matrix(action-1)
                memory.append((current_state, action_state, reward, next_state, done))
            
            if len(memory)>10000:
                memory.pop(0)
                
            if global_step!=0 and global_step>=1000 and env.frame==1 and episode%5==0:
                #print("Training started - Global_step {} Episode {}".format(global_step, episode))
                for epoch in range(0,10):
                    minibatch = random.sample(memory,32)
                    state_batch = [data[0] for data in minibatch]
                    action_batch = [data[1] for data in minibatch]
                    reward_batch = [data[2] for data in minibatch]
                    nextState_batch = [data[3] for data in minibatch]
                    terminal_batch = [data[4] for data in minibatch]
                    y_batch =[]
                    Qvalue_batch = sess.run(q, feed_dict={image_input:nextState_batch})
                    for i in range(32):
                        terminal = minibatch[i][4]
                        if terminal:
                            y_batch.append(reward_batch[i])
                        else:
                            y_batch.append(reward_batch[i]+0.95*np.max(Qvalue_batch[i]))
                    _, loss = sess.run([trainStep,cost],feed_dict={yInput:y_batch, actionInput:action_batch, image_input:state_batch})
                    epoch+=1
            reward_sum+=reward
            observation = next_observation
            env.frame+=1
            global_step+=1
            if done or env.frame>51:
                episode+=1
                print('Episode {} Reward {} Steps {}'.format(episode,reward_sum, env.frame))
                if episode==190:
                    save_path = saver.save(sess, "/tmp/model.ckpt")
                    print("Model saved in path: %s" % save_path)
                break