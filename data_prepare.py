import numpy as np
from tqdm import tqdm

data = np.load('./data2.npy')
gt = np.load('./gt2.npy')

data = np.transpose(data,(0,2,3,4,1))

print(data.shape) #(75, 155, 240, 240, 4)
print(gt.shape)  #(75, 155, 240, 240)
print(data.dtype)  #float32
print(gt.dtype) #uint8

# import matplotlib.pyplot as plt
# plt.imshow(data[5,70,:,:,2])
# plt.show()
# plt.imshow(gt[5,70,:,:])
# plt.show()


# As all slices does not show tumour region so only mid-portion i.e. 90th - 120th slice was taken to create final data
# each data is also cropped to centre with final dimension of (N1,192,192,4)
data = data[:,30:120,30:222,30:222,:].reshape([-1,192,192,4])
gt = gt[:,30:120,30:222,30:222].reshape([-1,192,192,1])

print(data.shape)  #(6750, 192, 192, 4)
print(gt.shape)  #(6750, 192, 192, 1)


#  GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
gt[np.where(gt==4)]=3   #converting ground truth value of 4 to 3 to do one hot encoding
# (Consider value 3 in results in output as class 4)


#  the data was randomly split into training, validation and test data with 60%:20%:20%
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, gt, test_size=0.20, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25,random_state=42)

# print("After split X_train, Y_train shape",X_train.shape, Y_train.shape)
# print("After split X_val, Y_val shape",X_val.shape, Y_val.shape)
# print("After split X_test, Y_test shape",X_test.shape, Y_test.shape)

# After split X_train, Y_train shape (4050, 192, 192, 4) (4050, 192, 192, 1)
# After split X_val, Y_val shape (1350, 192, 192, 4) (1350, 192, 192, 1)
# After split X_test, Y_test shape (1350, 192, 192, 4) (1350, 192, 192, 1)


# If your training data uses classes as numbers, to_categorical will transform those numbers in proper vectors for using with models
from keras.utils import to_categorical
Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)
X_train = (X_train-np.mean(X_train))/np.max(X_train)
X_test = (X_test-np.mean(X_test))/np.max(X_test)
X_val = (X_val-np.mean(X_val))/np.max(X_val)

# print("X_train, Y_train shape",X_train.shape, Y_train.shape)
# print("X_val, Y_val shape",X_val.shape, Y_val.shape)
# print("X_test, Y_test shape",X_test.shape, Y_test.shape)
#
# X_train, Y_train shape (4050, 192, 192, 4) (4050, 192, 192, 4)
# X_val, Y_val shape (1350, 192, 192, 4) (1350, 192, 192, 4)
# X_test, Y_test shape (1350, 192, 192, 4) (1350, 192, 192, 1)


# plt.imshow(X_train[1,:,:,1])
# plt.show()
# plt.imshow(Y_train[1,:,:,3])
# plt.show()
#
# np.save('./Training Data/X_train4.npy',X_train)
# np.save('./Training Data/Y_train4.npy',Y_train)
# np.save('./Validation Data/X_val4.npy',X_val)
# np.save('./Validation Data/Y_val4.npy',Y_val)
# np.save('./Test Data/X_test4.npy',X_test)
# np.save('./Test Data/Y_test4.npy',Y_test)

# print("Data saved successfully")