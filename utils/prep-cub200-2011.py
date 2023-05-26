from pathlib import Path
import pandas as pd
import tarfile

# raw_data = Path.home() / 'fgvc' / 'data' / 'CUB_200_2011.tgz'
# tar = tarfile.open(raw_data)
# tar.extractall()
# tar.close()


# read each meta data txt
classes = pd.read_csv(Path.cwd().parent / 'CUB_200_2011' / 'classes.txt', sep = ' ', names = ['Category', 'ClassName'])
labels = pd.read_csv(Path.cwd().parent / 'CUB_200_2011' / 'image_class_labels.txt', sep = ' ', names = ['ImageID', 'Category'])
images = pd.read_csv(Path.cwd().parent / 'CUB_200_2011' / 'images.txt', sep = ' ', names = ['ImageID', 'filename'])
splitting = pd.read_csv(Path.cwd().parent / 'CUB_200_2011' / 'train_test_split.txt', sep = ' ', names = ['ImageID', 'is_training'])

# join for full data
data = classes.merge(labels, on = 'Category')
data = data.merge(splitting, on = 'ImageID')
data = data.merge(classes, how = 'left', on = 'Category')
data['Label'] = data.Category - 1

# split to train and test set
train = data.loc[data['is_training'] == 1]
test = data.loc[data['is_training'] == 0]
print(train.shape, test.shape)

# save for csv file
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)