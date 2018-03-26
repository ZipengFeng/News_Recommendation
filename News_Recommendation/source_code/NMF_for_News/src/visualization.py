import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
norm_mAP = np.loadtxt('./mAP.txt')
cold_mAP = np.loadtxt('./mAP_cold.txt')
#train_sample_num = np.loadtxt('./mAP_train_smaple_num.txt')

#mAP = (np.sum(norm_mAP)+np.sum(cold_mAP))/(len(norm_mAP)+len(norm_mAP))
print(np.max(norm_mAP))

#norm_P_at_K = np.loadtxt('./P@K.txt')
#cold_P_at_K = np.loadtxt('./P@K_cold.txt')

norm_RecRank = np.loadtxt('./RecRank.txt')
cold_RecRank = np.loadtxt('./RecRank_cold.txt')
print(np.min(norm_RecRank))
print(np.min(cold_RecRank))
mAP = (np.sum(norm_RecRank)+np.sum(cold_RecRank))/(len(cold_RecRank)+len(norm_RecRank))
print(mAP)

#similarity_matrix = np.loadtxt('./similarity_matrix.txt')

# bins = np.arange(0, 100, 1)
# #plt.xlim([0, ])
# plt.ylim([0, 500])

#plt.hist(train_sample_num, bins=bins, alpha=0.7, label='old user')
#plt.hist(cold_P_at_K, bins=bins, alpha=0.5, label='new user')
#plt.title('Training sample number')
#plt.xlabel('Training sample number')
#plt.ylabel('number')
#plt.grid(True)
#plt.legend()
#plt.show()

#plt.scatter(train_sample_num, norm_mAP,s=25,alpha=0.4,marker='o')
#plt.title('AP along with sample number.')
#plt.xlabel('AP')
#plt.ylabel('Sample Number')
#plt.show()

# train_file = pd.read_csv("./train_data.txt", sep='\t', header=-1)
# test_file = pd.read_csv("./test_data.txt", sep='\t', header=-1)

# train_sample = set(train_file.iloc[:, 0])
# test_sample = set(test_file.iloc[:, 0])

# sample = list(train_sample & test_sample)
# training_list = []
# testing_list = []

# train_list = list(train_file.iloc[:, 0])
# test_list = list(test_file.iloc[:, 0])

# for i in range(len(sample)):
    # training_n = train_list.count(sample[i])
    # test_n = test_list.count(sample[i])
    # training_list.append(training_n)
    # testing_list.append(test_n)

# #plt.scatter(training_list, testing_list, s=25,alpha=0.4,marker='o')
# plt.hist(training_list, bins=bins, alpha=0.7, label='train')
# plt.hist(testing_list, bins=bins, alpha=0.7, label='test')
# plt.title('Sample number distribution')
# plt.xlabel('sample number')
# plt.ylabel('number')
# plt.legend()
# plt.grid(True)
# plt.show()

#print(training_list)

