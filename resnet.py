import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import model
import util
import time


class ResNet:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.net = model.Net().cuda() if self.use_cuda else model.Net()
        self.optimizer = None
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.train_time = []
        self.test_time = []
        self.start_epoch = 1

    def train(self, save_dir, num_epochs=75, batch_size=256, learning_rate=0.001, test_each_epoch=False, verbose=False):
        """Trains the network.

        Parameters
        ----------
        save_dir : str
            The directory in which the parameters will be saved
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        learning_rate : float
            The learning rate
        test_each_epoch : boolean
            True: Test the network after every training epoch, False: no testing
        verbose : boolean
            True: Print training progress to console, False: silent mode
        """
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
        self.net.train()

        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # train_transform = transforms.Compose([
        #     util.Cutout(num_cutouts=2, size=8, p=0.8),
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # ])

        train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=train_transform)
        # train_dataset = datasets.CIFAR10('data/cifar', train=True, download=True, transform=train_transform)
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        criterion = torch.nn.CrossEntropyLoss().cuda() if self.use_cuda else torch.nn.CrossEntropyLoss()

        progress_bar = util.ProgressBar()

        for epoch in range(self.start_epoch, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))

            epoch_correct = torch.tensor([0.]).to(self.device)
            epoch_total = 0
            epoch_loss = torch.tensor([0.]).to(self.device)
            epoch_start = time.time()
            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net.forward(images)
                loss = criterion(outputs, labels.squeeze_())
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    epoch_loss += loss
                    # print(f'loss:{loss.item()}')
                    _, predicted = torch.max(outputs.data, dim=1)
                    batch_total = labels.size(0)
                    batch_correct = (predicted == labels.flatten()).sum()

                    epoch_total += batch_total
                    epoch_correct += batch_correct

                if verbose:
                    # Update progress bar in console
                    info_str = 'Last batch accuracy: {:.4f} - Running epoch accuracy {:.4f}'.\
                                format(batch_correct.item() / batch_total, epoch_correct.item() / epoch_total)
                    progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)

            self.train_accuracies.append(epoch_correct.item() / epoch_total)
            self.train_losses.append(epoch_loss.item() / len(data_loader))
            self.train_time.append(time.time() - epoch_start)
            if verbose:
                progress_bar.new_line()

            if test_each_epoch:
                test_accuracy, test_time = self.test()
                self.test_accuracies.append(test_accuracy)
                self.test_time.append(test_time)
                if verbose:
                    print('Test accuracy: {}'.format(test_accuracy))

            # Save parameters after every epoch
            self.save_parameters(epoch, directory=save_dir)

    def test(self, batch_size=256, dataset='mnist'):
        """Tests the network.

        """
        self.net.eval()

        if dataset.lower() == 'mnist':
            test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                                 ])
            test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=test_transform)
        elif dataset.lower() == 'cifar':
            test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                 transforms.Grayscale(num_output_channels=1),
                                                 ])
            test_dataset = datasets.CIFAR10('data/cifar', train=False, download=True, transform=test_transform)
        else:
            raise Exception(f"Unknown dataset: {dataset}")

        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        correct = torch.tensor([0.]).to(self.device)
        total = 0
        start = time.time()
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels.flatten()).sum()

        self.net.train()
        return correct.item() / total, (time.time() - start) / len(data_loader)

    def save_parameters(self, epoch, directory):
        """Saves the parameters of the network to the specified directory.

        Parameters
        ----------
        epoch : int
            The current epoch
        directory : str
            The directory to which the parameters will be saved
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'train_losses': self.train_losses,
            'train_time': self.train_time,
            'test_time': self.test_time
        }, os.path.join(directory, 'resnet_' + str(epoch) + '.pth'))

    def load_parameters(self, path):
        """Loads the given set of parameters.

        Parameters
        ----------
        path : str
            The file path pointing to the file containing the parameters
        """
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001)
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_accuracies = checkpoint['test_accuracies']
        self.train_losses = checkpoint['train_losses']
        self.train_time = checkpoint['train_time']
        self.test_time = checkpoint['test_time']
        self.start_epoch = checkpoint['epoch']


import numpy as np
from resnet_numpy import resnet9_numpy

class ResNet_numpy:

    def __init__(self):
        self.device = torch.device('cpu')
        self.net = resnet9_numpy()
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.train_time = []
        self.test_time = []
        self.start_epoch = 1

    def train(self, save_dir, num_epochs=75, batch_size=256, learning_rate=0.001, test_each_epoch=False, verbose=False):
        """Trains the network.

        Parameters
        ----------
        save_dir : str
            The directory in which the parameters will be saved
        num_epochs : int
            The number of epochs
        batch_size : int
            The batch size
        learning_rate : float
            The learning rate
        test_each_epoch : boolean
            True: Test the network after every training epoch, False: no testing
        verbose : boolean
            True: Print training progress to console, False: silent mode
        """
        torch.autograd.set_detect_anomaly(True)
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)
        # self.net.train()

        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=train_transform)
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        # criterion = torch.nn.CrossEntropyLoss().cuda() if self.use_cuda else torch.nn.CrossEntropyLoss()

        progress_bar = util.ProgressBar()

        for epoch in range(self.start_epoch, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            epoch_correct = 0
            epoch_total = 0
            epoch_loss = 0
            epoch_start = time.time()
            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.to(self.device).numpy()
                labels = labels.to(self.device).numpy()

                # self.optimizer.zero_grad()
                outputs = self.net.forward(images)
                one_hot_labels = np.eye(10)[labels]
                loss = np.sum(-one_hot_labels * np.log(outputs)-(1-one_hot_labels) * np.log(1 - outputs)) / batch_size
                epoch_loss += loss
                out_diff_tensor = (outputs - one_hot_labels) / outputs / (1 - outputs) / batch_size
                # import ipdb; ipdb.set_trace()
                self.net.backward(out_diff_tensor, learning_rate)
                # print(f'loss:{loss}')


                # loss = criterion(outputs, labels.squeeze_())
                # loss.backward()
                # self.optimizer.step()

                predicted = np.argmax(outputs, axis=1)
                batch_total = labels.size
                batch_correct = (predicted == labels).sum()

                epoch_total += batch_total
                epoch_correct += batch_correct

                if verbose:
                    # Update progress bar in console
                    info_str = 'Last batch accuracy: {:.4f} - Running epoch accuracy {:.4f}'.\
                                format(batch_correct / batch_total, epoch_correct / epoch_total)
                    progress_bar.update(max_value=len(data_loader), current_value=i, info=info_str)

            self.train_accuracies.append(epoch_correct / epoch_total)
            self.train_losses.append(epoch_loss / len(data_loader))
            self.train_time.append(time.time() - epoch_start)
            if verbose:
                progress_bar.new_line()

            if test_each_epoch:
                test_accuracy, test_time = self.test()
                self.test_accuracies.append(test_accuracy)
                self.test_time.append(test_time)
                if verbose:
                    print('Test accuracy: {}'.format(test_accuracy))

            # Save parameters after every epoch
            # self.save_parameters(epoch, directory=save_dir)
            import pickle
            with open(f'{save_dir}/resnet_numpy_{epoch}.pkl', 'wb') as f:
                pickle.dump(self, f)

    def test(self, batch_size=256, dataset='mnist'):
        """Tests the network.

        """
        # self.net.eval()

        if dataset.lower() == 'mnist':
            test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                                 ])
            test_dataset = datasets.MNIST('data/mnist', train=False, download=True, transform=test_transform)
        elif dataset.lower() == 'cifar':
            test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                 transforms.Grayscale(num_output_channels=1),
                                                 ])
            test_dataset = datasets.CIFAR10('data/cifar', train=False, download=True, transform=test_transform)
        else:
            raise Exception(f"Unknown dataset: {dataset}")

        data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        correct = 0
        total = 0
        start = time.time()
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                images, labels = dataimages, labels = data
                images = images.to(self.device).numpy()
                labels = labels.to(self.device).numpy()

                outputs = self.net.forward(images)
                predicted = np.argmax(outputs, axis=1)
                correct += (predicted == labels).sum()
                total += labels.size

        # self.net.train()
        return correct / total, (time.time() - start) / len(data_loader)


    def save_parameters(self, epoch, directory):
        """Saves the parameters of the network to the specified directory.

        Parameters
        ----------
        epoch : int
            The current epoch
        directory : str
            The directory to which the parameters will be saved
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'train_losses': self.train_losses,
            'train_time': self.train_time,
            'test_time': self.test_time
        }, os.path.join(directory, 'resnet_' + str(epoch) + '.pth'))

    def load_parameters(self, path):
        """Loads the given set of parameters.

        Parameters
        ----------
        path : str
            The file path pointing to the file containing the parameters
        """
        # self.optimizer = torch.optim.Adam(self.net.parameters())
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_accuracies = checkpoint['test_accuracies']
        self.train_losses = checkpoint['train_losses']
        self.train_time = checkpoint['train_time']
        self.test_time = checkpoint['test_time']
        self.start_epoch = checkpoint['epoch']
