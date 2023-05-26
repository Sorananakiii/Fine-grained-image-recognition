from pathlib import Path
import pandas as pd
import tarfile


## Run it for 1 time to extract tar

# raw_data = Path.home() / 'fgvc' / 'data' / 'fgvc-aircraft-2013b.tar.gz'
# tar = tarfile.open(raw_data)
# tar.extractall(path = path)
# tar.close()

Classes = pd.read_csv('fgvc-aircraft-2013b/data/variants.txt', names = ['Classes'])

print(Classes.shape)
name = Classes.Classes.values
names = dict([*enumerate(name)])
names = {v: k for k, v in names.items()}
names


train = pd.read_csv('fgvc-aircraft-2013b/data/images_variant_train.txt', names = ['oneline'])

train['filename'] = train.oneline.apply(lambda x: x[:7] + '.jpg')
train['Classes'] = train.oneline.apply(lambda x: x[8:])
train['Labels'] = train.Classes.map(names)
train.drop('oneline', axis=1, inplace=True)
train.head()


val = pd.read_csv('fgvc-aircraft-2013b/data/images_variant_val.txt', names = ['oneline'])

val['filename'] = val.oneline.apply(lambda x: x[:7] + '.jpg')
val['Classes'] = val.oneline.apply(lambda x: x[8:])
val['Labels'] = val.Classes.map(names)
val.drop('oneline', axis=1, inplace=True)
val.head()


test = pd.read_csv('fgvc-aircraft-2013b/data/images_variant_test.txt', names = ['oneline'])

test['filename'] = test.oneline.apply(lambda x: x[:7] + '.jpg')
test['Classes'] = test.oneline.apply(lambda x: x[8:])
test['Labels'] = test.Classes.map(names)
test.drop('oneline', axis=1, inplace=True)
test.head()


trainset = pd.concat([train, val])
trainset = trainset.sort_values(by=['Labels']).reset_index()
trainset.drop(['index'], axis=1, inplace=True)
print(trainset.shape, test.shape)
trainset.to_csv('train_v2.csv', index=False)
test.to_csv('test.csv', index=False)