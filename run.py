import sys

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import  logging
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from Model.ta_oem import TA_OEM
from config import parse_opts
from data_process.dataloader import get_loader, MMDataLoader
from utils import Metric, Orthorhombic_loss, CMD_loss, Reconstruction_loss, Cos_loss, loadSaveData

logging.set_verbosity_error()
#logger = logging.getLogger('MSA')
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

def load_mosi_mosei_data(config, batch_data):
    """加载 MOSI/MOSEI 数据集的数据"""
    (sentences, visual, vlens, acoustic, alens, label, lengths,
     bert_sentences, bert_sentence_types, bert_sentence_att_mask,
     ids, v_masks, a_masks) = batch_data

    return {
        'visual': visual.to(config.device),
        'acoustic': acoustic.to(config.device),
        'sentences': sentences.to(config.device),
        'bert_sentences': bert_sentences.to(config.device),
        'bert_sentence_types': bert_sentence_types.to(config.device),
        'bert_sentence_att_mask': bert_sentence_att_mask.to(config.device),
        'label': label.to(config.device),
        'lengths': lengths,
        'indexes': ids,
        'vlens': vlens.to(config.device),
        'alens': alens.to(config.device),
        'v_masks': v_masks.to(config.device),
        'a_masks': a_masks.to(config.device),
    }

def load_sims_data(config, batch_data):
    """加载 SIMS 数据集的数据"""
    label = batch_data['labels']
    for k in label.keys():
        label[k] = label[k].to(config.device).view(-1, 1)

    return {
        'visual': batch_data['vision'].to(config.device),
        'acoustic': batch_data['audio'].to(config.device),
        'sentences': batch_data['text'].to(config.device),
        'label': label,
        'lengths': batch_data['audio_lengths'],
        'indexes': batch_data['id'],
    }

def load_iemocap_data(config, batch_data):
    """加载 SIMS 数据集的数据"""
    (batch_x, batch_y) = batch_data
    (id, text, audio, vision) = batch_x
    label = batch_y.squeeze(-1).long().view(-1)
    lenths = [20 for _ in range(text.shape[0])]

    return {
        'visual': vision.to(config.device),
        'acoustic': audio.to(config.device),
        'sentences': text.to(config.device),
        'label': label.to(config.device),
        'lengths': lenths,
        'indexes': id,
    }



class Automate():
    def __init__(self, args):
        self.args = args
        self.args.tasks = "MTAV"
        self.usgm = self.args.use_usgm
        self.metrics = Metric().getMetics(args.datasetName)
        self.criterion = nn.L1Loss(reduction='mean')
        self.criterion_clss = nn.CrossEntropyLoss()
        self.orthorhombic_loss = Orthorhombic_loss()
        self.cmd_loss = CMD_loss()
        self.reconstruction_loss = Reconstruction_loss(args)
        self.cos_loss = Cos_loss()
        self.feature_map = {
            'fusion': torch.zeros(args.train_samples, args.post_fusion_dim, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, args.post_text_dim, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, args.post_audio_dim, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, args.post_video_dim, requires_grad=False).to(args.device),
        }

        self.center_map = {
            'fusion': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'text': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'audio': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            },
            'vision': {
                'pos': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
                'neg': torch.zeros(args.post_fusion_dim, requires_grad=False).to(args.device),
            }
        }

        #First Verification
        self.center_label = {
            'pos': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'neg': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
        }

        self.center_label_map = {
            'fusion': {
                'pos': torch.zeros(1, requires_grad=False).to(args.device),
                'neg': torch.zeros(1, requires_grad=False).to(args.device),
            },
            'text': {
                'pos': torch.zeros(1, requires_grad=False).to(args.device),
                'neg': torch.zeros(1, requires_grad=False).to(args.device),
            },
            'audio': {
                'pos': torch.zeros(1, requires_grad=False).to(args.device),
                'neg': torch.zeros(1, requires_grad=False).to(args.device),
            },
            'vision': {
                'pos': torch.zeros(1, requires_grad=False).to(args.device),
                'neg': torch.zeros(1, requires_grad=False).to(args.device),
            }
        }

        self.dim_map = {
            'fusion': torch.tensor(args.post_fusion_dim).float(),
            'text': torch.tensor(args.post_text_dim).float(),
            'audio': torch.tensor(args.post_audio_dim).float(),
            'vision': torch.tensor(args.post_video_dim).float(),
        }
        # new labels
        self.label_map = {
            'fusion': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'text': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'audio': torch.zeros(args.train_samples, requires_grad=False).to(args.device),
            'vision': torch.zeros(args.train_samples, requires_grad=False).to(args.device)
        }

        self.name_map = {
            'M': 'fusion',
            'T': 'text',
            'A': 'audio',
            'V': 'vision'
        }

    def _forward_mosi_mosei(self, model, prepared_data):
        """MOSI/MOSEI 模型前向传播"""
        return model(
            text=prepared_data['sentences'],
            acoustic=prepared_data['acoustic'],
            visual=prepared_data['visual'],
            lenth=prepared_data['lengths'],
            bert_sent=prepared_data['bert_sentences'],
            bert_sent_type=prepared_data['bert_sentence_types'],
            bert_sent_mask=prepared_data['bert_sentence_att_mask']
        )


    def _forward_sims(self, model, prepared_data):
        """SIMS 模型前向传播"""
        return model(
            text=prepared_data['sentences'],
            acoustic=prepared_data['acoustic'],
            visual=prepared_data['visual'],
            lenth=prepared_data['lengths']
        )

    def _forward_iemocap(self, model, prepared_data):
        """IEMOCAP 模型前向传播"""
        return model(
            text=prepared_data['sentences'],
            acoustic=prepared_data['acoustic'],
            visual=prepared_data['visual'],
            lenth=prepared_data['lengths']
        )

    def _compute_loss(self, outputs, label, indexes=None):
        loss = 0
        ort_loss = self.orthorhombic_loss(outputs['fea_a'], outputs['fea_v'], outputs['fea_t'])
        ort_loss2 = (self.orthorhombic_loss(outputs['fea_a'], outputs['fea_m']) + self.orthorhombic_loss(
            outputs['fea_v'], outputs['fea_m']) + self.orthorhombic_loss(outputs['fea_t'], outputs['fea_m'])) / 3.0
        ort_loss3 = (self.orthorhombic_loss(outputs['fea_a'], outputs['fea_m_2']) + self.orthorhombic_loss(
            outputs['fea_v'], outputs['fea_m_2']) + self.orthorhombic_loss(outputs['fea_t'], outputs['fea_m_2'])) / 3.0
        recon_loss = (self.reconstruction_loss(outputs['ori_a'], outputs['rebu_a']) + self.reconstruction_loss(
            outputs['ori_v'], outputs['rebu_v']) + self.reconstruction_loss(outputs['ori_t'], outputs['rebu_t'])) / 3.0
        loss += 0.4 * ort_loss + 0.4 * ort_loss2 + 0.4 * ort_loss3 + 0.2 * recon_loss

        if self.args.use_usgm:
            assert self.args.datasetName in ['mosi', 'mosei', 'sims']
            assert indexes is not None
            for m in self.args.tasks:
                loss += self.weighted_loss(outputs[m], self.label_map[self.name_map[m]][indexes],
                                           indexes=indexes, mode=self.name_map[m])
        else:
            if self.args.datasetName == 'sims':
                for m in self.args.tasks:
                    loss += self.criterion(outputs[m], label[m])
            elif self.args.datasetName in ['mosi', 'mosei']:
                for m in self.args.tasks:
                    loss += self.criterion(outputs[m], label)
            elif self.args.datasetName == 'iemocap':
                loss += self.criterion_clss(outputs['M'].view(-1, 2), label)
        return loss

    def run(self, model, dataloader):

        if self.args.datasetName in ['mosi', 'mosei']:
            process_func = load_mosi_mosei_data
        elif self.args.datasetName == 'sims':
            process_func = load_sims_data
        else: #IEMOCAP
            process_func = load_iemocap_data

        if self.args.save_logs:
            writer = SummaryWriter(log_dir='logs', comment='Linear')


        bert_params = list(model.bertmodel.named_parameters())
        bert_params_decay = [p for n, p in bert_params]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'bertmodel' not in n]
        print('bert_params:', sum(p.numel() for p in bert_params_decay))
        print('model_params:', sum(p.numel() for p in model_params_other))

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert,
             'lr': self.args.learning_rate_bert},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other,
             'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.sch_list,
                                                         gamma=self.args.gamma)
        model = model.to(self.args.device)

        if self.args.datasetName in ['mosi', 'mosei']:
            forward_func = self._forward_mosi_mosei
            if self.usgm:
                for i, (sentences, visual, vlens, acoustic, alens, label, lengths, bert_sentences, bert_sentence_types,
                        bert_sentence_att_mask, ids, v_masks, a_masks) in enumerate(dataloader['train']):
                    labels_m = label.to(self.args.device)
                    indexes = ids.unsqueeze(1)
                    self.init_labels(indexes, labels_m)
        elif self.args.datasetName == 'sims':
            forward_func = self._forward_sims
        else:
            forward_func = self._forward_iemocap

        def train(model, dataloader, epoch):
            model.train()
            sum_loss = 0.0
            grad_clip_value = 0.8
            results = []
            truths = []

            for batch_data in tqdm(dataloader, leave=False):
                model.zero_grad()
                prepared_data = process_func(self.args, batch_data)
                indexes = prepared_data['indexes']
                label = prepared_data['label']
                outputs = forward_func(model, prepared_data)
                prads_avt = outputs['M']
                loss = self._compute_loss(outputs, label, indexes)

                # # # backward
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
                optimizer.step()

                # add features
                sum_loss += loss.item()
                if self.args.datasetName == 'iemocap':
                    results.append(prads_avt.view(-1, 2))
                    truths.append(label)
                elif self.args.datasetName in ['mosi', 'mosei']:
                    results.append(prads_avt)
                    truths.append(label)
                else:  #self.args.datasetName == 'sims'
                    results.append(prads_avt)
                    truths.append(label['M'])
                # update features
                if self.usgm:
                    # self.update_centers()
                    f_fusion = outputs['fea_m'].contiguous().view(outputs['fea_m'].size(0), -1).detach()
                    f_text = outputs['fea_t'].contiguous().view(outputs['fea_t'].size(0), -1).detach()
                    f_audio = outputs['fea_a'].contiguous().view(outputs['fea_a'].size(0), -1).detach()
                    f_vision = outputs['fea_v'].contiguous().view(outputs['fea_v'].size(0), -1).detach()
                    if epoch > 1:
                        self.update_labels(f_fusion, f_text, f_audio, f_vision, epoch, indexes, label)
                    self.update_features(f_fusion, f_text, f_audio, f_vision, indexes)
                    self.update_centers()

            #self.update_centers()
            sum_loss = sum_loss / len(dataloader)
            results = torch.cat(results)
            truths = torch.cat(truths)
            return sum_loss, results, truths

        def test(model, dataloader, epoch):
            model.eval()
            sum_loss = 0.0
            grad_clip_value = 0.8
            results = []
            truths = []
            with torch.no_grad():
                for batch_data in tqdm(dataloader):
                    model.zero_grad()
                    prepared_data = process_func(self.args, batch_data)
                    indexes = prepared_data['indexes']
                    label = prepared_data['label']
                    outputs = forward_func(model, prepared_data)
                    prads_avt = outputs['M']
                    loss = self._compute_loss(outputs, label, indexes)

                    # add features
                    sum_loss += loss.item()
                    if self.args.datasetName == 'iemocap':
                        results.append(prads_avt.view(-1, 2))
                        truths.append(label)
                    elif self.args.datasetName in ['mosi', 'mosei']:
                        results.append(prads_avt)
                        truths.append(label)
                    else:  # self.args.datasetName == 'sims'
                        results.append(prads_avt)
                        truths.append(label['M'])

                sum_loss = sum_loss / len(dataloader)
                results = torch.cat(results)
                truths = torch.cat(truths)
                return sum_loss, results, truths


        min_mae = 100.0
        best_epoch = 0
        for epoch in range(1, self.args.epoch):
            tra_loss, tra_results, tra_truths = train(model, dataloader['train'], epoch=epoch)
            val_loss, val_results, val_truths = test(model, dataloader['dev'], epoch=epoch)
            test_loss, test_results, test_truths = test(model, dataloader['test'], epoch=epoch)

            print('epoch:', epoch)
            print('train loss:', tra_loss)
            print('train')
            mae_train = self.metrics(tra_results, tra_truths)
            print('eval')
            mae_dev = self.metrics(val_results, val_truths)
            print('test')
            mae_test = self.metrics(test_results, test_truths)
            print("*" * 50)
            scheduler.step()

            if self.args.save_logs:
                writer.add_scalar('train_loss',tra_loss, epoch)
                writer.add_scalar('val_loss', val_loss, epoch)
                writer.add_scalar('test_loss', test_loss, epoch)

            # save best model
            if self.args.save_best_model:
                folder_path = self.args.model_save_dir
                if mae_dev < min_mae:
                    min_mae = mae_dev
                    os.makedirs(folder_path, exist_ok=True)
                    file_path = os.path.join(folder_path, self.args.file_name)
                    torch.save(model.state_dict(), file_path)
                    best_epoch = epoch
                    self.best_file_name = file_path

            # # early stop
            # if epoch - best_epoch >= self.args.early_stop and self.args.early_stop:
            #     print('\n early stop in :', epoch)
            #     try:
            #         model = loadSaveData(model, self.best_file_name)
            #         _, results, truths = test(dataloader['test'], epoch=epoch)
            #         self.metrics(results, truths)
            #     except Exception as e:
            #         print('load model false')
            #     break



    def practical_apply(self, model, data):
        model = model.to(self.args.device)
        model.eval()
        acoustic = data['audio'].to(self.args.device)
        visual = data['visual'].to(self.args.device)
        sentences = data['text'].to(self.args.device)
        lengths = data['lenths']

        outputs = model(sentences, acoustic, visual, lengths,)
        preds_avt = outputs['M']
        print('predicted sentiment:', preds_avt)


    def weighted_loss(self, y_pred, y_true, indexes=None, mode='fusion'):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        if mode == 'fusion':
            weighted = torch.ones_like(y_pred)
        else:
            weighted = torch.tanh(torch.abs(self.label_map[mode][indexes] - self.label_map['fusion'][indexes]))
        loss = torch.mean(weighted * torch.abs(y_pred - y_true))
        return loss

    def init_labels(self, indexes, m_labels):
        self.label_map['fusion'][indexes] = m_labels
        self.label_map['text'][indexes] = m_labels
        self.label_map['audio'][indexes] = m_labels
        self.label_map['vision'][indexes] = m_labels

    def update_features(self, f_fusion, f_text, f_audio, f_vision, indexes):
        self.feature_map['fusion'][indexes] = f_fusion
        self.feature_map['text'][indexes] = f_text
        self.feature_map['audio'][indexes] = f_audio
        self.feature_map['vision'][indexes] = f_vision

    def update_centers(self):
        def update_single_center(mode):
            neg_indexes = self.label_map[mode] < 0
            if self.args.excludeZero:
                pos_indexes = self.label_map[mode] > 0
            else:
                pos_indexes = self.label_map[mode] >= 0
            self.center_map[mode]['pos'] = torch.mean(self.feature_map[mode][pos_indexes], dim=0)
            self.center_map[mode]['neg'] = torch.mean(self.feature_map[mode][neg_indexes], dim=0)
            # self.center_label_map[mode]['pos'] = torch.mean(self.label_map[mode][pos_indexes], dim=0)
            # self.center_label_map[mode]['neg'] = torch.mean(self.label_map[mode][neg_indexes], dim=0)

        update_single_center(mode='fusion')
        update_single_center(mode='text')
        update_single_center(mode='audio')
        update_single_center(mode='vision')

    def update_labels(self, f_fusion, f_text, f_audio, f_vision, cur_epoches, indexes, outputs):
        MIN = 1e-8

        def update_single_label(f_single, mode):
            d_sp = F.cosine_similarity(f_single, self.center_map[mode]['pos'], dim=-1)
            d_sn = F.cosine_similarity(f_single, self.center_map[mode]['neg'], dim=-1)
            # d_sp = torch.norm(f_single - self.center_map[mode]['pos'], dim=-1)
            # d_sn = torch.norm(f_single - self.center_map[mode]['neg'], dim=-1)
            # delta_s = ( d_sp-d_sn)/(torch.norm(f_single)*
            #        (torch.norm(self.center_map[mode]['pos'],dim=-1)-torch.norm(self.center_map[mode]['neg'],dim=-1))+MIN)
            delta_s = (d_sn - d_sp)
            alpha = delta_s / (delta_f + MIN)
            new_labels = 0.5 * alpha * self.label_map[mode][indexes] + \
                         0.5 * (self.label_map[mode][indexes] + delta_s - delta_f)
            new_labels = torch.clamp(new_labels, min=-self.args.H, max=self.args.H)
            # new_labels = torch.tanh(new_labels) * self.args.H
            n = cur_epoches
            self.label_map[mode][indexes] = (n - 1) / (n + 1) * self.label_map[mode][indexes] + 2 / (n + 1) * new_labels

        d_fp = F.cosine_similarity(f_fusion, self.center_map['fusion']['pos'], dim=-1)
        d_fn = F.cosine_similarity(f_fusion, self.center_map['fusion']['neg'], dim=-1)
        delta_f = (d_fn - d_fp)
        update_single_label(f_text, mode='text')
        update_single_label(f_audio, mode='audio')
        update_single_label(f_vision, mode='vision')

    def text_unimodality(self, model, dataloader):
        true_label_map = {
            'fusion': torch.zeros(self.args.train_samples, requires_grad=False),
            'text': torch.zeros(self.args.train_samples, requires_grad=False),
            'audio': torch.zeros(self.args.train_samples, requires_grad=False),
            'vision': torch.zeros(self.args.train_samples, requires_grad=False),
        }

        for i, data in enumerate(dataloader):
            label = data['labels']
            indexes = data['id']
            for k in label.keys():
                label[k] = label[k].view(-1)

            true_label_map['audio'][indexes] = label['A']
            true_label_map['vision'][indexes] = label['V']
            true_label_map['text'][indexes] = label['T']

        calu_label_t = self.label_map['text'].detach().cpu()
        calu_label_a = self.label_map['audio'].detach().cpu()
        calu_label_v = self.label_map['vision'].detach().cpu()


        def plot_comparison(calu_label, true_label, name):
            plt.figure(figsize=(8,6))
            calu_label = np.array(calu_label)
            true_label = np.array(true_label)

            plt.boxplot([true_label, calu_label], positions=[0,1],
                        labels=['True Label', 'Calculated Label'], patch_artist=True)

            colors = ['red', 'lightblue']
            for patch, color in zip(plt.gca().artists, colors):
                patch.set_facecolor(color)

            plt.xticks(fontsize=20)
            plt.ylabel('Value',fontsize=20)
            plt.tight_layout(pad=0.1)
            plt.savefig(self.args.res_save_dir + name)

        plot_comparison(calu_label_v, true_label_map['vision'],name='vis_label')
        plot_comparison(calu_label_a, true_label_map['audio'], name='aud_label')
        plot_comparison(calu_label_t, true_label_map['text'], name='tex_label')


    def train_loader(self, model):

        datasetName = self.args.datasetName
        assert datasetName in ['mosi', 'mosei', 'sims', 'iemocap']
        if datasetName in ['mosi', 'mosei', 'iemocap']:
            train_loader = get_loader(config, mode='train', shuffle=True)
            dev_loader = get_loader(config, mode='valid', shuffle=False)
            text_loader = get_loader(config, mode='test', shuffle=False)

            dataloader = {
                'train': train_loader,
                'dev': dev_loader,
                'test': text_loader,
            }
            if datasetName == 'iemocap':
                self.args.use_usgm = False

        elif datasetName == 'sims':

            data = MMDataLoader(config)

            dataloader = {
                'train': data['train'],
                'dev': data['valid'],
                'test': data['test'],
            }
        else:
            sys.exit(1)

        self.run(model, dataloader)

if __name__ == "__main__":
    logging.set_verbosity_error()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    config = parse_opts('mosi')  # [mosi, mosei, sims, iemocap]
    origin_file = os.path.dirname(os.path.abspath(__file__))
    do_model_file = os.path.join(origin_file, 'saves', 'save_model', 'save_do_model.pth')
    do_model = Automate(config)
    model = TA_OEM(config)
    do_model.train_loader(model)
    save_dict = {
        'label': do_model.label_map
    }
    torch.save(save_dict, do_model_file)


    # config = parse_opts('sims')
    # data = MMDataLoader(config)
    # dataloaded = {
    #     'train': data['train'],
    #     'valid': data['valid'],
    #     'test': data['test'],
    # }
    # model = TA_OEM(config)
    # do_model = Automate(config)
    # origin_file = os.path.dirname(os.path.abspath(__file__))
    # file = os.path.join(origin_file, 'saves', 'save_model', 'save_new_device')
    # do_model_file = os.path.join(origin_file, 'saves', 'save_model', 'save_do_model.pth')
    # model = loadSaveData(model,file)
    # save_dict = torch.load(do_model_file)
    # do_model.label_map = save_dict['label']
    # do_model.text_unimodality(model, dataloaded['train'])