import copy
import json

import pandas
import torch.nn.functional as F
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score, r2_score

from Model.ta_oem import TA_OEM
from config import parse_opts
from data_process.dataloader import get_loader
import  matplotlib.colors as mcolors

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))



class  Metric():
    def __init__(self):
        self.metrics_dict = {
            'MOSI': self.__eval_mosi_regression,
            'MOSEI': self.__eval_mosei_regression,
            'IEMOCAP': self.__eval_iemocap_classification,
            'SIMS': self.__eval_sims_regression,
        }

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.eval_mosei_senti(y_pred, y_true)

    def __eval_mosei_regression(self, y_pred, y_true):
        return self.eval_mosei_senti(y_pred, y_true)

    def __eval_iemocap_classification(self, y_pred, y_true):
        return self.eval_iemocap(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        return self.eval_sims_senti(y_pred, y_true)

    def eval_mosei_senti(self, results, truths):
        test_preds = results.view(-1).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

        f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
        binary_truth_non0 = test_truth[non_zeros] > 0
        binary_preds_non0 = test_preds[non_zeros] > 0
        f_score_non0 = f1_score(binary_truth_non0, binary_preds_non0, average='weighted')
        acc_2_non0 = accuracy_score(binary_truth_non0, binary_preds_non0)

        binary_truth_has0 = test_truth >= 0
        binary_preds_has0 = test_preds >= 0
        acc_2 = accuracy_score(binary_truth_has0, binary_preds_has0)
        f_score = f1_score(binary_truth_has0, binary_preds_has0, average='weighted')

        mult_a7 = mult_a7 * 100
        mult_a5 = mult_a5 * 100
        acc1 = np.round(acc_2, 4) * 100
        acc2 = np.round(acc_2_non0, 4) * 100
        f1_1 = np.round(f_score, 4) * 100
        f1_2 = np.round(f_score_non0, 4) * 100

        print("MAE: ", mae)
        print("Correlation Coefficient: ", corr)
        print("mult_acc_7: ", mult_a7)
        print("mult_acc_5: ", mult_a5)
        print("Accuracy all/non0: {}/{}".format(acc1, acc2))
        print("F1 score all/non0: {}/{} over {}/{}".format(f1_1, f1_2,
                                                           binary_truth_has0.shape[0], binary_truth_non0.shape[0]))
        return mae

    def eval_sims_senti(self, results, truths):
        test_preds = results.view(-1).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # weak sentiment two classes{[-0.6, 0.0], (0.0, 0.6]}
        ms_2 = [-1.01, 0.0, 1.01]
        weak_index_l = np.where(test_truth >= -0.4)[0]
        weak_index_r = np.where(test_truth <= 0.4)[0]
        weak_index = [x for x in weak_index_l if x in weak_index_r]
        test_preds_weak = test_preds[weak_index]
        test_truth_weak = test_truth[weak_index]
        test_preds_a2_weak = test_preds_weak.copy()
        test_truth_a2_weak = test_truth_weak.copy()
        for i in range(2):
            test_preds_a2_weak[np.logical_and(test_preds_weak > ms_2[i], test_preds_weak <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2_weak[np.logical_and(test_truth_weak > ms_2[i], test_truth_weak <= ms_2[i + 1])] = i

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)
        mult_a2_weak = self.__multiclass_acc(test_preds_a2_weak, test_truth_a2_weak)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')
        r2 = r2_score(test_truth, test_preds)

        eval_results = {
            "Mult_acc_2": mult_a2,
            "Mult_acc_2_weak": mult_a2_weak,
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,
            "F1_score": f_score,
            "MAE": mae,
            "Corr": corr,  # Correlation Coefficient
            "R_squre": r2
        }

        print("MAE: ", mae)
        print("Correlation Coefficient: ", corr)
        print("mult_acc_3: ", mult_a3)
        print("mult_acc_5: ", mult_a5)
        print("Accuracy_2 normal/week: {}/{}".format(mult_a2,mult_a2_weak))
        print("F1 score : {}".format(f_score))

        return mae

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def eval_iemocap(self, results, truths, single=-1):
        emos = ["Neutral", "Happy", "Sad", "Angry"]
        if single < 0:
            test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
            test_truth = truths.view(-1, 4).cpu().detach().numpy()

            for emo_ind in range(4):
                print(f"{emos[emo_ind]}: ")
                test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
                test_truth_i = test_truth[:, emo_ind]
                f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
                acc = accuracy_score(test_truth_i, test_preds_i)
                print(" - F1 Score: ", f1)
                print(" - Accuracy: ", acc)
        else:
            test_preds = results.view(-1, 2).cpu().detach().numpy()
            test_truth = truths.view(-1).cpu().detach().numpy()

            print(f"{emos[single]}: ")
            test_preds_i = np.argmax(test_preds, axis=1)
            test_truth_i = test_truth
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print(" - F1 Score: ", f1)
            print(" - Accuracy: ", acc)

        return 0


    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]




class Orthorhombic_loss(torch.nn.Module):
    def __init__(self):
        super(Orthorhombic_loss, self).__init__()
    def forward(self,res1,res2,res3=None):
        if res3 is not None:
            orthor_1_2 = self.calu_SOC(res2, res1)
            orthor_1_3 = self.calu_SOC(res3, res1)
            orthor_2_3 = self.calu_SOC(res3, res2)
            sum = (orthor_1_2 + orthor_1_3 + orthor_2_3)/3.0
            return sum
        else:
            orthor_invar_spec = self.calu_SOC(res2,res1)
            return orthor_invar_spec
    def calu_SOC(self,A,B):
        #A: torch.Size([B, 2, e_dim])
        product = torch.matmul(A,B.transpose(-1,-2))
        squared_frobenius_norm = torch.norm(product, p='fro', dim=(-2,-1))
        result_loss = torch.sum(squared_frobenius_norm,dim=0)
        #print('squared_frobenius_norm:',squared_frobenius_norm)
        return result_loss.item()


class CMD_loss(torch.nn.Module):
    def __init__(self):
        super(CMD_loss, self).__init__()

    def forward(self, res1, res2):
        cmd_12 = self.cmd(res1, res2)
        return  cmd_12

    def cmd(self, X, Y, order=2):
        """
        计算两个张量 X 和 Y 之间的 CMD 距离。

        参数:
        X (np.ndarray): 张量 X，形状为 (n_samples, n_features)。
        Y (np.ndarray): 张量 Y，形状为 (m_samples, n_features)。
        kernel (str): 核函数类型，默认为 'rbf'（高斯核）。
        gamma (float): 高斯核的参数 gamma，如果为 None，则使用 1/n_features。
        返回:
        float: CMD 距离值。
        """
        batch_size = X.size(0)
        X = X.contiguous().view(batch_size,-1)
        Y = Y.contiguous().view(batch_size, -1)
        mean1 = torch.mean(X, dim=0)
        mean2 = torch.mean(Y, dim=0)

        # 中心化张量
        centered1 = X - mean1
        centered2 = Y - mean2

        # 计算中心矩
        moment1 = torch.mean(centered1 ** order, dim=0)
        moment2 = torch.mean(centered2 ** order, dim=0)

        # 计算中心矩之间的距离
        cmd_dist = torch.norm(moment1 - moment2, p=2)
        return cmd_dist.item()


class Reconstruction_loss(torch.nn.Module):
    def __init__(self,config):
        super(Reconstruction_loss, self).__init__()
        self.config = config
    def forward(self,res_raw,res_new):
        recon_num = self.calu_rec(res_raw, res_new)
        return recon_num
    def calu_rec(self,A,B):
        batch_size = A.size(0)
        A = A.contiguous().view(batch_size, -1)
        B = B.contiguous().view(batch_size, -1)

        d = F.cosine_similarity(A, B)
        loss = torch.sum(d, dim=0)
        # norm_b_std = torch.std(B, dim=0)
        # norm_a_std = torch.std(A, dim=0)
        # norm_of_difference = torch.norm((A-B),p=2)
        return loss.item() / self.config.hidden_dim


class Cos_loss(torch.nn.Module):
    def __init__(self):
        super(Cos_loss, self).__init__()

    def forward(self, res1, res2):
        cmd_12 = self.cos_loss(res1, res2)
        return  cmd_12

    def cos_loss(self, X, Y):

        batch_size = X.size(0)
        X = X.contiguous().view(batch_size,-1)
        Y = Y.contiguous().view(batch_size, -1)
        d = F.cosine_similarity(X, Y)
        loss = torch.sum(d,dim=0)
        return loss.item()



def loadSaveData(model, recordFile):
    checkpoint = torch.load(recordFile)
    newModel = copy.deepcopy(model)  # 深拷贝模型结构
    newModel.load_state_dict(checkpoint)
    return newModel



def tSNE():
    data_set = 'mosei'
    config = parse_opts(data_set)
    train_loader = get_loader(config, mode='train', shuffle=True)
    dev_loader = get_loader(config, mode='valid', shuffle=False)
    text_loader = get_loader(config, mode='test', shuffle=False)
    dataloaded = {
        'train': train_loader,
        'dev': dev_loader,
        'test': text_loader,
    }
    model = TA_OEM(config)
    # notice: choose your parm file
    model = loadSaveData(model,'save/moesi_max_0.86')
    model.eval()
    results = []
    truths = []

    with torch.no_grad():
        for i, (sentences, visual, vlens, acoustic, alens, label, lengths, bert_sentences, bert_sentence_types,
                bert_sentence_att_mask, ids, v_masks, a_masks) in enumerate(text_loader):
            visual = visual.to(config.device)
            acoustic = acoustic.to(config.device)
            sentences = sentences.to(config.device)
            bert_sentences = bert_sentences.to(config.device)
            bert_sentence_types = bert_sentence_types.to(config.device)
            bert_sentence_att_mask = bert_sentence_att_mask.to(config.device)
            lengths = lengths
            outputs = model(sentences, acoustic, visual, lengths, bert_sentences, bert_sentence_types,
                            bert_sentence_att_mask)
            fea_m = outputs['fea_m']
            fea_m_2 = outputs['fea_m_2']
            fea_a = outputs['fea_a']
            fea_v = outputs['fea_v']
            fea_t = outputs['fea_t']

            results.append(fea_m)
            truths.append(0)

            results.append(fea_m_2)
            truths.append(1)

            results.append(fea_t)
            truths.append(2)

            results.append(fea_a)
            truths.append(3)

            results.append(fea_v)
            truths.append(4)


    features = np.concatenate(results, axis=0)
    labels = np.array(truths)
    tsne = TSNE(n_components=2, perplexity=150, learning_rate=0.5, n_iter=1000)
    reduced_features = tsne.fit_transform(features)


    hex_colors = [(67,136,112),(219,67,44),(131,138,175),(196,183,151),(255,255,0)]
    # 将十六进制颜色代码转换为RGB元组
    rgb_colors = [(r / 255, g / 255, b / 255) for r, g, b in hex_colors]

    Class = ["invariant_one", "invariant_two", "specific_t", "specific_a", "specific_v"]
    plt.figure(figsize=(10, 8))
    for i in range(5):
        plt.scatter(reduced_features[labels == i, 0], reduced_features[labels == i, 1], c=[rgb_colors[i]], label=Class[i])
    plt.legend(fontsize=16, loc='best')
    plt.xlabel('Feature 1', fontsize=20)
    plt.ylabel('Feature 2', fontsize=20)
    plt.tight_layout(pad=0)
    plt.savefig('saves/save_fig/feature_tsne')


def draw_cruse():
    df = pandas.read_csv('loss.csv')
    loss1 = df['loss1'].rolling(window=10).mean()
    loss2 = df['loss2'].rolling(window=10).mean()
    loss3 = df['loss3'].rolling(window=10).mean()


    # 绘制波形图
    plt.figure(figsize=(14, 8))
    hex_colors = [(67, 136, 112), (219, 67, 44), (131, 138, 175), (196, 183, 151)]
    rgb_colors = [(r / 255, g / 255, b / 255) for r, g, b in hex_colors]
    plt.plot(range(1, len(loss1) + 1), loss1, linestyle='-', color=rgb_colors[0], label='SFN(u1,u2)', lw=4)
    plt.fill_between(range(1, len(loss1) + 1),loss1-150,y2=loss1+150, alpha=0.2, color=rgb_colors[0])
    plt.plot(range(1, len(loss2) + 1), loss2, linestyle='-', color=rgb_colors[1], label='SFN(m1,u)',lw=4)
    plt.fill_between(range(1, len(loss2) + 1), loss2-150,y2=loss2+150, alpha=0.2, color=rgb_colors[1])
    plt.plot(range(1, len(loss3) + 1), loss3, linestyle='-', color=rgb_colors[2], label='SFN(m2,u)', lw=4)
    plt.fill_between(range(1, len(loss3) + 1), loss3-150,y2=loss3+150, alpha=0.2, color=rgb_colors[2])
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.gca().yaxis.set_tick_params(labelleft=False)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout(pad=0.2)
    plt.savefig('saves/save_fig/SFN_loss')



def tSNE_cmu_cluser():
    data_set = 'mosi'
    config = parse_opts(data_set)
    text_loader = get_loader(config, mode='test', shuffle=False)
    model = TA_OEM(config)
    model = model.to(config.device)
    # choose your parm file
    model = loadSaveData(model, 'save/save_model/mosi_epoch+2')
    model.eval()
    results = []
    truths = []

    for i, (sentences, visual, vlens, acoustic, alens, label, lengths, bert_sentences, bert_sentence_types,
            bert_sentence_att_mask, ids, v_masks, a_masks) in enumerate(text_loader):
        visual = visual.to(config.device)
        acoustic = acoustic.to(config.device)
        sentences = sentences.to(config.device)
        bert_sentences = bert_sentences.to(config.device)
        bert_sentence_types = bert_sentence_types.to(config.device)
        bert_sentence_att_mask = bert_sentence_att_mask.to(config.device)
        lengths = lengths
        outputs = model(sentences, acoustic, visual, lengths, bert_sentences, bert_sentence_types,
                        bert_sentence_att_mask)
        fea_m = outputs['fea_m'].contiguous().view(outputs['fea_m'].size(0), -1).detach().cpu().numpy()
        truths.append(label.cpu().numpy())
        results.extend(fea_m)

    print('results:', len(results))
    print('truths:', len(truths))
    features = np.array(results)
    labels = np.concatenate(truths,axis=0)
    print('features:',features.shape)
    print('labels:', labels.shape)


    tsne = TSNE(n_components=2, perplexity=150, learning_rate=0.6, n_iter=1500)
    reduced_features = tsne.fit_transform(features)
    cmap = plt.cm.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=-3,vmax=3)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(reduced_features[:,0], reduced_features[:,1], c=labels, cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, label='Label Value')
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Label Value', fontsize=20)
    plt.xlabel('Feature 1', fontsize=20)
    plt.ylabel('Feature 2', fontsize=20)
    plt.tight_layout(pad=0.2)
    plt.savefig('(mosi)2')