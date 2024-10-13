from resnet import ResNet_numpy
import pickle
 

net = ResNet_numpy()

# f=open(f'saves/epoch1.pkl','rb')  
# net.net=pickle.load(f)  


net.train(save_dir='saves', num_epochs=75, batch_size=256, learning_rate=0.01, verbose=True)
accuracy = net.test()
print('Test accuracy: {}'.format(accuracy))

# f.close()