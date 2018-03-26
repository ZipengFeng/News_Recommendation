import numpy as np
from scipy.sparse import csr_matrix
import math
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

class NMF:
    def __init__(self, user_news_df, dimk, idf_w, Lambda):
        self.dimk = dimk
        self.nmf = NMF(n_components=self.dimk, init='nndsvdar', solver='mu', tol=1, alpha=3, l1_ratio=0.3)
        self.ui_df = user_news_df
        self.USER_NUM = 10000
        self.ITEM_NUM = 6183
        self.COLD = 0
        self.Lambda = Lambda
        self.ui_mat = self.get_mat(user_news_df)  
        #self.user_sim = np.matmul(self.ui_mat, self.ui_mat.transpose())#cosine_similarity(self.ui_mat)
        #self.user_sim = normalize(self.user_sim, norm = 'l2')
        self.similarity_matrix = np.loadtxt("./Data/similarity_matrix.txt")
        #self.similarity_matrix *= idf_w
        #print(self.user_sim)

    def construct_ui_matrix(self, ui_df):
        read_sum = ui_df.shape[0]
        user_row = np.array([self.ui_df.iloc[i, 0] for i in range(read_sum)])
        item_col = np.array([self.ui_df.iloc[i, 1] for i in range(read_sum)])
        mat = np.zeros((self.USER_NUM, self.ITEM_NUM))
        for i in range(read_sum):
            mat[user_row[i], item_col[i]] += 1
        return mat

    def decomposition(self):  
        self.user_factors = self.nmf.fit_transform(self.ui_mat)
        self.item_factors = self.nmf.components_

    def predict_topK(self, user, L, Lambda):
        cold_start = True
        user_rating = self.ui_mat[user, :]
        predict_rating_list = self.predict_matrix_sim[user, :]
        rec_list = dict()
        for item in range(self.ITEM_NUM):
            if user_rating[item] == 0:
                if item in self.candidate_news:
                    rec_list[item] = predict_rating_list[item]
                else:
                    rec_list[item] = -1
            else:
                rec_list[item] = -1
                cold_start = False
        rec_topK = sorted(rec_list.items(), key=lambda e: e[1], reverse=True)
        return cold_start, [rec_topK[i][0] for i in range(L)]

    def evaluation(self, test_df, topK = 5):
        read_sum = test_df.shape[0]
        user_row = np.array([test_df.iloc[i, 0] for i in range(read_sum)])
        item_col = np.array([test_df.iloc[i, 1] for i in range(read_sum)])
        read_score = np.array([1 for i in range(read_sum)])
        self.test_mat = csr_matrix((read_score, (user_row, item_col)), shape=(self.USER_NUM, self.ITEM_NUM))
        self.predict_matrix = np.matmul(self.user_factors, self.item_factors)
        #self.predict_matrix_u_sim = np.matmul(self.user_sim, self.predict_matrix)
        self.predict_matrix_u_sim = 1-cosine_similarity(self.predict_matrix)#np.matmul(self.predict_matrix, self.predict_matrix.transpose())
        self.predict_matrix_u_sim = np.matmul(self.predict_matrix_u_sim, self.predict_matrix)
        self.predict_matrix_i_sim = np.matmul(self.predict_matrix, self.similarity_matrix) 
        self.predict_matrix_sim = self.predict_matrix_u_sim*(-self.Lambda) + self.predict_matrix_i_sim*self.Lambda        

        self.candidate_news = set(test_df.iloc[:, 1])
        ui_dict = dict()
        for i in range(test_df.shape[0]):
            if test_df.iloc[i, 0] not in ui_dict.keys():
                ui_dict[test_df.iloc[i, 0]] = [test_df.iloc[i, 1]]
            else:
                ui_dict[test_df.iloc[i, 0]].append(test_df.iloc[i, 1])

        mAP = []
        mAP_cold = []
        mPrecision = []
        mPrecision_cold = []
        mAP_train_sample_num = []
        mP_at_K = []
        mP_at_K_cold = []
        eval_user = 0
        user_sum = len(ui_dict)
        for user, itemlist in ui_dict.items():
            eval_user += 1
            cold_start, predlist = self.predict_topK(user, topK, self.Lambda)
            reclist = list(set(itemlist))
            if cold_start:
                mPrecision_cold.append(self.cal_PN(predlist, reclist, topK))
                mAP_cold.append(self.cal_AP(predlist, reclist))
                mP_at_K_cold.append(self.cal_P_at_K(predlist, reclist, K=10))
                self.COLD += 1
            else:
                mPrecision.append(self.cal_PN(predlist, reclist, topK))
                mAP.append(self.cal_AP(predlist, reclist))
                mAP_train_sample_num.append(np.sum(self.ui_mat[user, :]))
                mP_at_K.append(self.cal_P_at_K(predlist, reclist, K=10))
        mAP = np.array(mAP)
        mAP_cold = np.array(mAP_cold)
        mPrecision = np.array(mPrecision)
        mean_Precision = np.mean(mPrecision)
        max_Precision = np.max(mPrecision)
        min_Precision = np.min(mPrecision)
        median_Precision = np.median(mPrecision)
        min_mAP = np.min(mAP)
        max_mAP = np.max(mAP)
        mean_mAP = np.mean(mAP)
        mean_mAP_cold = np.mean(mAP_cold)
        median_mAP = np.median(mAP)
        mean_P_at_K = np.mean(np.array(mP_at_K))
        mean_P_at_K_cold = np.mean(np.array(mP_at_K_cold))
        mean_P_at_K_all = int(mean_P_at_K*len(mP_at_K)+mean_P_at_K_cold*len(mP_at_K_cold))/user_sum
        print("COLD START %d" % self.COLD)
        print("Top%d Rec Result:" % topK)
        print("mean RecRank: %g  mAP: %g   mAP_cold: %g" % (mean_Precision, mean_mAP, mean_mAP_cold))
        print("median RecRank: %g  mAP: %g" % (median_Precision, median_mAP))
        print("min RecRank: %g  mAP: %g" % (min_Precision, min_mAP))
        print("max RecRank: %g  mAP: %g" % (max_Precision, max_mAP))
        print("mean P@K: %g  P@K COLD: %g" % (mean_P_at_K, mean_P_at_K_cold))
        print("P@K %g" % mean_P_at_K_all)
        np.savetxt("./Data/mAP.txt", mAP)
        np.savetxt("./Data/RecRank.txt", mPrecision)
        np.savetxt("./Data/mAP_train_smaple_num.txt", mAP_train_sample_num)
        np.savetxt("./Data/mAP_cold.txt", mAP_cold)
        np.savetxt("./Data/RecRank_cold.txt", mPrecision_cold)
        np.savetxt("./Data/P@K.txt", mP_at_K)
        np.savetxt("./Data/P@K_cold.txt", mP_at_K_cold)

    def cal_PN(self, predlist, reclist, n=10):
        p = 0
        for pred in predlist:
            p += 1
            if pred in reclist:
                return p
        return p
    
    def cal_P_at_K(self, predlist, reclist, K=10):
        p = 0
        for i in range(K):
            if predlist[i] in reclist:
                p += 1
        return p/K
    
    def cal_AP(self, predlist, reclist):
        pos = 1
        rel = 1
        ap = 0
        for i in range(len(predlist)):
            if predlist[i] in reclist:
                ap += rel / pos
                rel += 1
            pos += 1
        ap /= len(reclist)
        return ap
