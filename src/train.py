%load_ext autoreload
%autoreload 2

import torch, cv2, time
from torchvision import transforms
import pandas as pd
from PIL import Image
from src.datasets import FGVSDataset
from src.model import *
from src.loss import ArcMargin


# Configuration
CUDA_LAUNCH_BLOCKING = 1
TORCH_USE_CUDA_DSA = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
pd.set_option('display.max_colwidth', 200)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)
])


trainset = FGVSDataset('train.csv', "CUB_200_2011/images", transform = train_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True)

testset = FGVSDataset('test.csv', "CUB_200_2011/images", transform = test_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False)


tune_lr = 0.0001
base_lr = 0.001
weight_decay = 5e-4
num_classes = 200

F_Extraction = BaseModel('resnet50').to(device)
F_Connected = Dense().to(device)
Arcface = ArcMargin(512, num_classes, s=64, m1=0.5).to(device)

criterion = nn.CrossEntropyLoss()

Extraction_opt = torch.optim.SGD(F_Extraction.parameters(), lr=tune_lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
Extraction_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Extraction_opt, mode='max', factor=0.1, patience=4, min_lr=1e-7, verbose=True)

fullycon_opt = torch.optim.SGD(F_Connected.parameters(), lr=base_lr, momentum=0.9, nesterov=True)
fullycon_scheduler = torch.optim.lr_scheduler.MultiStepLR(fullycon_opt, milestones=[30,50], gamma=0.1)

Arcface_opt = torch.optim.SGD(Arcface.parameters(), lr=base_lr, momentum=0.9, nesterov=True)
Arcface_scheduler = torch.optim.lr_scheduler.MultiStepLR(Arcface_opt, milestones=[30,50], gamma=0.1)


epochs = 2
save_model_path = 'Checkpoint'
steps = 0
running_loss = 0

print('Start fine-tuning...')
best_acc = 0.
best_epoch = None
end_patient = 0

for epoch in range(epochs):
    
    start_time = time.time()
    for idx, data in enumerate(train_loader):
        steps += 1
        
        # Move input and label tensors to the default device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        Extraction_opt.zero_grad()
        fullycon_opt.zero_grad()
        Arcface_opt.zero_grad()
        
        feature = F_Extraction(inputs)
        feature = F_Connected(feature)
        _, output = Arcface(feature, labels)
        loss = criterion(output, labels)

        loss.backward()
        Extraction_opt.step()
        fullycon_opt.step()
        Arcface_opt.step()

        running_loss += loss.item()
    stop_time = time.time()
    print('Epoch {}/{} and used time: {:.4f} sec.'.format(epoch+1, epochs, stop_time - start_time))
    
    F_Extraction.eval(), F_Connected.eval(), Arcface.eval()
    for name, loader in [("train", train_loader), ("test", test_loader)]:
        _acc = 0
        _loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in loader:
                
                imgs, labels = data
                imgs, labels = imgs.to(device), labels.to(device)
                
                feature = F_Extraction(imgs)
                feature = F_Connected(feature)
                logit, output = Arcface(feature, labels)
                loss = criterion(output, labels)
                _loss += loss.item()
                
                result = F.softmax(logit, dim=1)
                _, predicted = torch.max(result, dim=1)
                
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
            _acc = 100 * correct  / total
            _loss = _loss / len(loader)
            
        print('{} loss: {:.6f}    {} accuracy: {:.4f}'.format(name, _loss, name, _acc))
    print()
    

    running_loss = 0
    F_Extraction.train(), F_Connected.train(), Arcface.train()
    Extraction_scheduler.step(_acc)
    fullycon_scheduler.step(_acc)
    Arcface_scheduler.step(_acc)
    
    if _acc > best_acc:
#         model_file = os.path.join(save_model_path, 'resnet34_CUB_200_fine_tuning_epoch_{}_acc_{}.pth'.format(best_epoch, best_acc))
        
#         if os.path.isfile(model_file):
#             os.remove(os.path.join(save_model_path, 'resnet34_CUB_200_fine_tuning_epoch_{}_acc_{}.pth'.format(best_epoch, best_acc)))
        
        end_patient = 0
        best_acc = _acc
        best_epoch = epoch + 1
#         print('The accuracy is improved, save model')
#         torch.save(model.state_dict(), os.path.join(save_model_path,'resnet34_CUB_200_fine_tuning_epoch_{}_acc_{}.pth'.format(best_epoch, best_acc)))
        
    else:
        end_patient += 1

    if end_patient >= 10 and epoch > 50:
        break
print('After the training, the end of the epoch {}, the highest accuracy is {:.2f}'.format(best_epoch, best_acc))