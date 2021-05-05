import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pytorch_model_summary import summary
import argparse

from pybullet_real2sim.supervised_learning.image_generator import DataGenerator
from pybullet_real2sim.image_transforms import *
from pybullet_real2sim.plot_tools import *
from pybullet_real2sim.common_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-use_gpu", type=str2bool, default=True)
args = parser.parse_args()

# mode (gpu vs direct for comparison)
use_gpu = args.use_gpu
if use_gpu:
    device = 'cuda'
else:
    device = 'cpu'

train_data_dir = '../data_collection/sim/data/edge2dTap/rigid/256x256/square_360_-6_6_train'
# validation_data_dir = '../data_collection/sim/data/edge2dTap/rigid/256x256/square_360_-6_6_val'

# generated dataset
# validation_data_dir = '../data_collection/generated_data/edge2dTap/square_360_-6_6_val/64_250epoch_[-6,6]_specnorm'
# validation_data_dir = '../data_collection/generated_data/edge2dTap/square_360_-6_6_val/128_250epoch_[-6,6]_specnorm'
validation_data_dir = '../data_collection/generated_data/edge2dTap/square_360_-6_6_val/256_250epoch_[-6,6]_specnorm'

# train_data_dir = '../data_collection/real/data/edge2dTap/square_360_-6_6_train'
# validation_data_dir = '../data_collection/real/data/edge2dTap/square_360_-6_6_val'

# Parameters
params = {'batch_size':  32,
          'epochs':      100,
          'lr':          1e-4,
          'lr_factor':   0.5,
          'lr_patience': 8,
          'dropout':     0.25,
          'dim':         (128,128),
          'bbox':        [70,0,550,480] if 'real' in train_data_dir else None,
          'shuffle':     True,
          'rshift':      (0.05, 0.02),
          'rzoom':       (0.95, 1),
          'thresh':      True if 'real' in train_data_dir else False,
          'brightlims':  None,  #[0.3,1.0,-50,50], # alpha limits for contrast, beta limits for brightness
          'noise_var':   None,  # 0.001,
          'stdiz':       False,
          'normlz':      True,
          'train_data_dir':    os.path.join(train_data_dir, 'images'),
          'train_target_file': os.path.join(train_data_dir, 'targets.csv'),
          'val_data_dir':      os.path.join(validation_data_dir,  'images'),
          'val_target_file':   os.path.join(validation_data_dir,  'targets.csv')
          }

training_generator = DataGenerator(target_file=params['train_target_file'],
                                   data_dir=params['train_data_dir'],
                                   dim=params['dim'],
                                   bbox=params['bbox'],
                                   stdiz=params['stdiz'],
                                   normlz=params['normlz'],
                                   thresh=params['thresh'],
                                   rshift=params['rshift'],
                                   rzoom=params['rzoom'],
                                   brightlims=params['brightlims'],
                                   noise_var=params['noise_var'])

val_generator = DataGenerator(target_file=params['val_target_file'],
                              data_dir=params['val_data_dir'],
                              dim=params['dim'],
                              bbox=params['bbox'],
                              stdiz=params['stdiz'],
                              normlz=params['normlz'],
                              thresh=params['thresh'],
                              rshift=None,
                              rzoom=None,
                              brightlims=None,
                              noise_var=None)

n_workers = 32
training_loader = torch.utils.data.DataLoader(training_generator,
                                              batch_size=params['batch_size'],
                                              shuffle=params['shuffle'],
                                              num_workers=n_workers)

val_loader = torch.utils.data.DataLoader(val_generator,
                                         batch_size=params['batch_size'],
                                         shuffle=params['shuffle'],
                                         num_workers=n_workers)

n_train_batches = len(training_loader)
n_val_batches   = len(val_loader)


# Define network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # input_channels = 1, output_channels = 16
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=11, stride=1, padding=5)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # input channels = 16, output_channels = 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=9, stride=1, padding=4)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # input channels = 32, output_channels = 32
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # # input channels = 32, output_channels = 32
        self.conv4 = nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # input_neurons = 16*25*25, output_neurons = 256
        self.fc1 = nn.Linear(32 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool4(F.relu(self.conv4_bn(self.conv4(x))))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=params['dropout'])
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=params['dropout'])
        x = self.fc3(x)
        return x

model = Net().to(device)
print(summary(Net(), torch.zeros((1, 1, *params['dim'])), show_input=True))


def acc_metric(labels, predictions):
    # predictions and labels as array for easier use
    MSE = torch.abs(labels-predictions)

    # separate into r and theta
    r_diff_array = MSE[:,0]
    theta_diff_array = MSE[:,1]

    # find the values below a threshold
    r_correct = r_diff_array<0.5
    theta_correct = theta_diff_array<2
    both_correct = r_correct & theta_correct

    # count the number correct for accuracy
    total_correct = torch.sum(both_correct)
    total_counted = predictions.shape[0]

    return total_correct.cpu().numpy(), total_counted


# define optimizer and loss
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=params['lr'])
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=params['lr_factor'], patience=params['lr_patience'], verbose=True)

#Time for printing
training_start_time = time.time()

# for tracking metrics across training
train_loss_list = []
train_acc_list = []
validation_loss_list = []
validation_acc_list = []

#Loop for n_epochs
for epoch in range(params['epochs']):

    print_every = (n_train_batches // 10)
    start_time = time.time()
    running_loss = 0.0
    epoch_train_loss = 0.0
    total_train_correct = 0
    total_train_counted = 0

    for i, train_batch in enumerate(training_loader, 0):

        #Get inputs
        inputs, labels = train_batch['images'], train_batch['labels']

        #Wrap them in a Variable object
        inputs, labels = Variable(inputs).float().to(device), Variable(labels).float().to(device)

        #Set the parameter gradients to zero
        optimizer.zero_grad()

        #Forward pass, backward pass, optimize
        outputs = model(inputs)
        loss_size = loss(outputs, labels)
        loss_size.backward()
        optimizer.step()

        # count correct for accuracy metric
        correct, counted = acc_metric(labels, outputs)
        total_train_correct += correct
        total_train_counted += counted

        #Print statistics
        running_loss += loss_size.item()
        epoch_train_loss += loss_size.item()

        #Print every 10th batch of an epoch
        if (i + 1) % (print_every + 1) == 0 or (i+1) % n_train_batches == 0:
            percentage_complete = int(100 * (i+1) / n_train_batches)
            print("Epoch {},    complete: {:d}%    time taken: {:.2f}s    train_loss: {:.2f}    train_acc: {:.2f} ".format(
                    epoch+1, percentage_complete, time.time() - start_time, running_loss / print_every, total_train_correct/total_train_counted), end="\n" if percentage_complete==100 else '\r')

            #Reset running loss and time
            running_loss = 0.0

    # append training loss and acc
    train_loss_list.append(epoch_train_loss/n_train_batches)
    train_acc_list.append(total_train_correct/total_train_counted)

    #At the end of the epoch, do a pass on the validation set
    model.eval() # turn off batchnorm/dropout

    total_val_loss = 0
    total_val_correct = 0
    total_val_counted = 0
    for val_batch in val_loader:

        #Get inputs
        inputs, labels = val_batch['images'], val_batch['labels']

        #Wrap tensors in Variables
        inputs, labels = Variable(inputs).float().to(device), Variable(labels).float().to(device)

        #Forward pass
        val_outputs = model(inputs)
        val_loss_size = loss(val_outputs, labels)
        total_val_loss += val_loss_size.item()

        # count correct for accuracy metric
        correct, counted = acc_metric(labels, val_outputs)
        total_val_correct += correct
        total_val_counted += counted

    # display and track metrics
    val_loss = total_val_loss / n_val_batches
    val_acc  = total_val_correct / total_val_counted

    validation_loss_list.append(val_loss)
    validation_acc_list.append(val_acc)
    print("val_loss = {:.2f},     val_acc: {:.2f}".format(val_loss, val_acc), end="\n")

    # turn back to training model
    model.train()

    # decay the lr
    lr_scheduler.step(val_loss)
    # print('lr: {:.8f}'.format(optimizer.param_groups[0]['lr']))

print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

# run through "test" set (validation for now)
full_label_list = []
full_pred_list = []
for val_batch in val_loader:

    #Get inputs
    inputs, labels = val_batch['images'], val_batch['labels']

    #Wrap tensors in Variables
    inputs, labels = Variable(inputs).to(device), Variable(labels).float().to(device)

    #Forward pass
    val_outputs = model(inputs)

    # predictions and labels as array for easier use
    pred_array = val_outputs.cpu().detach().numpy()
    label_array = labels.cpu().detach().numpy()
    full_label_list.append(label_array)
    full_pred_list.append(pred_array)

# plot progress
plot_training(train_loss_list, validation_loss_list, train_acc_list, validation_acc_list)

# convert into array
full_label_array = np.concatenate( full_label_list, axis=0 )
full_pred_array = np.concatenate( full_pred_list, axis=0 )
plot_radial_error(full_label_array, full_pred_array)
