from pathlib import Path
import pandas as pd
import tarfile, os

path = 'devkit'
dirs = os.listdir(path)
print(dirs)


# Default variables
annotation = defaultdict()

# list annotation
for filename in dirs:
    if filename[-4:] == '.mat':
        annotation[filename[:-4]] = scipy.io.loadmat(os.path.join(path, filename))
#         print(annotation[filename[:-4]].shape)


def get_labels(cars_meta, train=True):
    
    if train:
        annotations = annotation['cars_train_annos']['annotations'][0]
    else:
        annotations = annotation['cars_test_annos_withlabels']['annotations'][0]
        
    classes = annotation[cars_meta]['class_names'][0]
    class_names = dict(zip(range(1, len(classes)),[c[0] for c in classes]))
    
    labelled_images = {}
    dataset = []
    for i,arr in enumerate(annotations):
        # the last entry in the row is the image name
        # The rest is the data, first bbox, then classid
        dataset.append([y[0][0] for y in arr][0:5]+[arr[5][0]])
    # Convert to a DataFrame, and specify the column names
    temp_df = pd.DataFrame(dataset, 
                           columns =['BBOX_X1','BBOX_Y1','BBOX_X2','BBOX_Y2','ClassID','filename'])

    temp_df = temp_df.assign(ClassName = temp_df.ClassID.map(dict(class_names)))
    temp_df.columns = ['bbox_x1','bbox_y1','bbox_x2','bbox_y2','Category','filename', 'class_name']
    return temp_df




train_df = get_labels('cars_meta')
train_df['Labels'] = train_df.Category - 1
train_df.to_csv('train.csv', index=False)

test_df = get_labels('cars_meta', train=False)
test_df['Labels'] = test_df.Category - 1
test_df.to_csv('test.csv', index=False)
# # Add missing class name! - 'smart fortwo Convertible 2012'
# train_df.loc[train_df['class_name'].isnull(), 'class_name'] = 'smart fortwo Convertible 2012'
# test_df.loc[test_df['class_name'].isnull(), 'class_name'] = 'smart fortwo Convertible 2012'

# frames = [train_df, test_df]
# labels_df = pd.concat(frames)
# labels_df.reset_index(inplace=True, drop=True)
# labels_df = labels_df[['filename', 'bbox_x1', 'bbox_y1','bbox_x2','bbox_y2',
#                             'class_id', 'class_name','is_test']]

# # adjust the test file names
# labels_df['filename'].loc[labels_df['is_test']==1] = 'test_' + labels_df['filename']

# # Add the cropped file names
# labels_df['filename_cropped'] = labels_df['filename'].copy()
# labels_df['filename_cropped'].loc[labels_df['is_test']==0] = 'cropped_' + labels_df['filename']

# labels_df.to_csv(path + 'labels_with_annos.csv')
# labels_df.head()


print('training set has shape: ', train_df.shape)
train_df.head()

print('test set has shape: ', test_df.shape)
test_df.head()