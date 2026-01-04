import sys

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from transformers import BertConfig, BertModel, BertTokenizer

from Model.att import my_encoder


def att_encoder(config, q, v):
    return my_encoder(d_emb_q=q, d_emb_v=v, n_head=config.n_heads,
                                       d_ff=config.d_ff,
                                       dropout=config.drop, d_k=config.d_k, d_v=config.d_v,
                                       n_position=config.batch_size)

def classifier(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False, drop=0.5):
    return nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv1d(in_dim // 2, in_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Dropout(drop),
            nn.LeakyReLU(),
            nn.Conv1d(in_dim // 2, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.AdaptiveAvgPool1d(1)
        )


class TA_OEM(nn.Module):

    def __init__(self, config): #guide='T'/'V'/'A'
        super(TA_OEM, self).__init__()
        self.config = config
        self.device = config.device
        self.guide = guide = config.guide
        e_dim = config.hidden_dim
        num_classes = config.num_classes
        self.input_size = hidden_sizes = config.input_size
        bert_path = config.bert_path_en
        self.batch_first = batch_first = config.batch_first
        # feature extract
        bertconfig = BertConfig.from_pretrained(bert_path, output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained(bert_path, config=bertconfig)
        self.text_model_sims = BertTextEncoder(config, language='cn', use_finetune=False)

        #batch_first = False
        self.arnn1 = nn.LSTM(self.input_size[0], hidden_sizes[0], bidirectional=True, batch_first=batch_first)
        self.arnn2 = nn.LSTM(2 * hidden_sizes[0], hidden_sizes[0], bidirectional=True, batch_first=batch_first)

        self.vrnn1 = nn.LSTM(self.input_size[1], hidden_sizes[1], bidirectional=True, batch_first=batch_first)
        self.vrnn2 = nn.LSTM(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True, batch_first=batch_first)

        self.trnn1 = nn.LSTM(self.input_size[2], hidden_sizes[2], bidirectional=False, batch_first=batch_first)
        self.trnn2 = nn.LSTM(hidden_sizes[2], hidden_sizes[2], bidirectional=False, batch_first=batch_first)

        self.alayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))
        self.tlayer_norm = nn.LayerNorm((hidden_sizes[2],))

        self.linear_tex = nn.Linear(2*hidden_sizes[2], hidden_sizes[2],bias=False)
        if self.config.datasetName in ['mosi', 'mosei']:
            e_dim_a = 4 * hidden_sizes[0]
            e_dim_v = 4 * hidden_sizes[1]
            e_dim_t = hidden_sizes[2]
        elif self.config.datasetName == 'sims':
            e_dim_a = hidden_sizes[0]
            e_dim_v = hidden_sizes[1]
            e_dim_t = hidden_sizes[2]
        else:
            e_dim_a = 4 * hidden_sizes[0]
            e_dim_v = 4 * hidden_sizes[1]
            e_dim_t = hidden_sizes[2]
        #modality share
        assert guide in ['T', 'A', 'V']
        if guide == 'T':
            self.my_trans_t_a = att_encoder(config, q=e_dim_t, v=e_dim_a)
            self.my_trans_t_v = att_encoder(config, q=e_dim_t, v=e_dim_v)
            self.my_trans_ta_tv = att_encoder(config, q=e_dim_t, v=e_dim_t)
            self.my_trans_tv_ta = att_encoder(config, q=e_dim_t, v=e_dim_t)
            self.my_trans_a = att_encoder(config, q=e_dim_a, v=e_dim_a)
            self.my_trans_v = att_encoder(config, q=e_dim_v, v=e_dim_v)
            self.my_trans_t = att_encoder(config, q=e_dim_t, v=e_dim_t)

            # modaility align
            self.aligned_a = nn.Sequential(
                nn.Linear(e_dim_a, e_dim, bias=False),
                nn.ReLU())

            self.aligned_v = nn.Sequential(
                nn.Linear(e_dim_v, e_dim, bias=False),
                nn.ReLU())

            self.aligned_t = nn.Sequential(
                nn.Linear(e_dim_t, e_dim, bias=False),
                nn.ReLU())

            self.project_ta_tv = nn.Sequential(
                nn.Linear(2 * e_dim_t, e_dim, bias=False),
                nn.ReLU())

            # share modality in AT and AV
            self.project_tav = nn.Sequential(
                nn.Linear(2 * e_dim_t, e_dim, bias=False),
                nn.ReLU())

        elif guide == 'V':
            self.my_trans_t_a = att_encoder(config, q=e_dim_v, v=e_dim_a)
            self.my_trans_t_v = att_encoder(config, q=e_dim_v, v=e_dim_t)
            self.my_trans_ta_tv = att_encoder(config, q=e_dim_v, v=e_dim_v)
            self.my_trans_tv_ta = att_encoder(config, q=e_dim_v, v=e_dim_v)
            self.my_trans_a = att_encoder(config, q=e_dim_a, v=e_dim_a)
            self.my_trans_v = att_encoder(config, q=e_dim_t, v=e_dim_t)
            self.my_trans_t = att_encoder(config, q=e_dim_v, v=e_dim_v)

            # modaility align
            self.aligned_a = nn.Sequential(
                nn.Linear(e_dim_a, e_dim, bias=False),
                nn.ReLU())

            self.aligned_v = nn.Sequential(
                nn.Linear(e_dim_t, e_dim, bias=False),
                nn.ReLU())

            self.aligned_t = nn.Sequential(
                nn.Linear(e_dim_v, e_dim, bias=False),
                nn.ReLU())

            self.project_ta_tv = nn.Sequential(
                nn.Linear(2 * e_dim_v, e_dim, bias=False),
                nn.ReLU())

            # share modality in AT and AV
            self.project_tav = nn.Sequential(
                nn.Linear(2 * e_dim_v, e_dim, bias=False),
                nn.ReLU())

        else:
            self.my_trans_t_a = att_encoder(config, q=e_dim_a, v=e_dim_t)
            self.my_trans_t_v = att_encoder(config, q=e_dim_a, v=e_dim_v)
            self.my_trans_ta_tv = att_encoder(config, q=e_dim_a, v=e_dim_a)
            self.my_trans_tv_ta = att_encoder(config, q=e_dim_a, v=e_dim_a)
            self.my_trans_a = att_encoder(config, q= e_dim_t, v= e_dim_t)
            self.my_trans_v = att_encoder(config, q=e_dim_v, v=e_dim_v)
            self.my_trans_t = att_encoder(config, q=e_dim_a, v=e_dim_a)

            # modaility align
            self.aligned_a = nn.Sequential(
                nn.Linear(e_dim_t, e_dim, bias=False),
                nn.ReLU())

            self.aligned_v = nn.Sequential(
                nn.Linear(e_dim_v, e_dim, bias=False),
                nn.ReLU())

            self.aligned_t = nn.Sequential(
                nn.Linear(e_dim_a, e_dim, bias=False),
                nn.ReLU())

            self.project_ta_tv = nn.Sequential(
                nn.Linear(2 * e_dim_a, e_dim, bias=False),
                nn.ReLU())

            # share modality in AT and AV
            self.project_tav = nn.Sequential(
                nn.Linear(2 * e_dim_a, e_dim, bias=False),
                nn.ReLU())


        # notice: only T
        self.MLP_Communicator1 = MLP_Communicator(hidden_sizes[2], 2, hidden_size=e_dim, depth=1)
        self.MLP_Communicator2 = MLP_Communicator(hidden_sizes[2], 2, hidden_size=e_dim, depth=1)



        self.rebuild_project_a = nn.Sequential(
            nn.Linear(e_dim, e_dim_a, bias=False),
            nn.ReLU())

        self.rebuild_project_v = nn.Sequential(
            nn.Linear(e_dim, e_dim_v, bias=False),
            nn.ReLU())

        self.rebuild_project_t = nn.Sequential(
            nn.Linear(e_dim, e_dim_t, bias=False),
            nn.ReLU())

        # classifier
        self.cls_token = nn.Parameter(torch.zeros(config.batch_size, 1, e_dim))  # (32,1,256)
        self.pos_embed = nn.Parameter(torch.zeros(config.batch_size, 2 + 2*2 + 3*2,e_dim))  # (32,4,256)

        self.classific_a = classifier(in_dim=e_dim, out_dim=num_classes, drop=config.drop)
        self.classific_v = classifier(in_dim=e_dim, out_dim=num_classes, drop=config.drop)
        self.classific_t = classifier(in_dim=e_dim, out_dim=num_classes, drop=config.drop)
        self.classific_m = classifier(in_dim=e_dim, out_dim=num_classes, drop=config.drop)


    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths,batch_first=self.batch_first)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1,batch_first=self.batch_first)  #
        normed_h1 = layer_norm(padded_h1)  #
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths,batch_first=self.batch_first)  #
        _, (final_h2, _) = rnn2(packed_normed_h1)  # final_h2.shape = ([2, 32, 35])
        return final_h1, final_h2  # final_h1.shape = ([2, 32, 35]) final_h2.shape = ([2, 32, 35]

    def shared_modaties(self, utterance_a, utterance_v, utterance_t):
        # Projecting to same sized space
        # utterance_a: torch.Size([B, 296])
        # utterance_v: torch.Size([B, 188])
        # utterance_t: torch.Size([B, 768])
        a_1,_ = self.my_trans_a(torch.cat([torch.unsqueeze(utterance_a, 0),torch.unsqueeze(utterance_a, 0)],dim=0))    #torch.Size([2, 32, 768])
        self.specific_a = self.aligned_a(a_1).permute(1, 0, 2)
        v_1,_ = self.my_trans_v(torch.cat([torch.unsqueeze(utterance_v, 0),torch.unsqueeze(utterance_v, 0)],dim=0))    #torch.Size([2, 32, 768])
        self.specific_v = self.aligned_v(v_1).permute(1, 0, 2)
        t_1,_ = self.my_trans_t(torch.cat([torch.unsqueeze(utterance_t, 0),torch.unsqueeze(utterance_t, 0)],dim=0))    #torch.Size([2, 32, 768])
        self.specific_t = self.aligned_t(t_1).permute(1, 0, 2)



        self.invariant_t_a,_ = self.my_trans_t_a(torch.cat([torch.unsqueeze(utterance_t, 0), torch.unsqueeze(utterance_t, 0)], dim=0),
                                             torch.cat([torch.unsqueeze(utterance_a, 0), torch.unsqueeze(utterance_a, 0)], dim=0))    #torch.Size([2, 32, 768])
        self.invariant_t_v,_ = self.my_trans_t_v(torch.cat([torch.unsqueeze(utterance_t, 0), torch.unsqueeze(utterance_t, 0)], dim=0),
                                             torch.cat([torch.unsqueeze(utterance_v, 0), torch.unsqueeze(utterance_v, 0)], dim=0))   #torch.Size([2, 32, 768])
        self.invariant_ta_tv,_ = self.my_trans_ta_tv(self.invariant_t_a,self.invariant_t_v) #torch.Size([2, B, 768])
        self.invariant_tv_ta,_ = self.my_trans_tv_ta(self.invariant_t_v,self.invariant_t_a) #torch.Size([2, B, 768])
        # self.invariant_ta_tv = self.MLP_Communicator1(self.invariant_ta_tv.permute(1,0,2))
        # self.invariant_tv_ta = self.MLP_Communicator2(self.invariant_tv_ta.permute(1,0,2))
        self.invariant_feature = torch.cat([self.invariant_ta_tv.permute(1,0,2),self.invariant_tv_ta.permute(1,0,2)],dim=2) #torch.Size([B, 2, 768*2])
        self.hie_att_two = torch.cat([self.invariant_t_a,self.invariant_t_v],dim=2).permute(1,0,2)


    def forward(self,text, acoustic, visual, lenth, bert_sent=None, bert_sent_type=None, bert_sent_mask=None):
        # cmu
        # text: torch.Size([B, 27])
        # audio: torch.Size([B, 27, 47])
        # vision: torch.Size([B, 27, 74])
        assert self.config.datasetName in ['mosi', 'mosei', 'iemocap', 'sims']
        if self.config.datasetName == 'sims':
            batch_size = text.shape[0]
            tex = self.text_model_sims(text)[:, 0, :]
            aud = acoustic
            vis = visual

        elif self.config.datasetName in ['mosi', 'mosei']:
            batch_size = lenth.shape[0]
            bert_output = self.bertmodel(input_ids=bert_sent, attention_mask=bert_sent_mask,token_type_ids=bert_sent_type)  # bert_sent_type ([64, 41])
            bert_output = bert_output[0]

            masked_output = torch.mul(bert_sent_mask.unsqueeze(2),
                                      bert_output)  # ([64, 41])->([64, 41, 1]) @ ([64, 41, 768]) -> masked_output ([64, 41, 768])
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  # mask_len (64,1)
            bert_output = torch.sum(masked_output, dim=1,keepdim=False) / mask_len  # 'torch.cuda.FloatTensor' bert_output ([64, 768])
            tex = bert_output  # ([64, 768])

            final_h1a, final_h2a = self.extract_features(acoustic, lenth, self.arnn1, self.arnn2, self.alayer_norm)
            aud = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
            # final_h1a = final_h2a=[2,32,47] aud=[32,188]
            final_h1v, final_h2v = self.extract_features(visual, lenth, self.vrnn1, self.vrnn2, self.vlayer_norm)
            vis = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        else:
            batch_size = text.shape[0]
            final_h1t, final_h2t = self.extract_features(text, lenth, self.trnn1, self.trnn2, self.tlayer_norm)
            tex = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
            tex = self.linear_tex(tex)

            final_h1a, final_h2a = self.extract_features(acoustic, lenth, self.arnn1, self.arnn2, self.alayer_norm)
            aud = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
            # final_h1a = final_h2a=[2,32,47] aud=[32,188]
            final_h1v, final_h2v = self.extract_features(visual, lenth, self.vrnn1, self.vrnn2, self.vlayer_norm)
            vis = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
                # final_h1v = final_h2v=[2,32,74] vis=[32,296]

        aud_ori = torch.cat([torch.unsqueeze(aud, dim=1), torch.unsqueeze(aud, dim=1)], dim=1)
        vis_ori = torch.cat([torch.unsqueeze(vis, dim=1), torch.unsqueeze(vis, dim=1)], dim=1)
        tex_ori = torch.cat([torch.unsqueeze(tex, dim=1), torch.unsqueeze(tex, dim=1)], dim=1)
        if self.guide == 'T':
            self.shared_modaties(aud, vis, tex)
        elif self.guide == 'V':
            self.shared_modaties(aud, tex, vis)
        else:
            self.shared_modaties(tex, vis, aud)
        #self.invariant_feature = self.MLP_Communicator1(self.invariant_feature)

        # self.specific_a torch.Size([B, 2, e_dim])
        invariant_feature = self.project_ta_tv(self.invariant_feature) #torch.Size([B, 2, e_dim])
        invariant_feature_two = self.project_tav(self.hie_att_two)  #torch.Size([B, 2, e_dim])

        rebuild_a = self.rebuild_project_a(self.specific_a)
        rebuild_v = self.rebuild_project_v(self.specific_v)
        rebuild_t = self.rebuild_project_t(self.specific_t)

        a = self.classific_a(self.specific_a.permute(0,2,1)).contiguous().view(batch_size,-1)
        v = self.classific_v(self.specific_v.permute(0,2,1)).contiguous().view(batch_size,-1)
        t = self.classific_t(self.specific_t.permute(0,2,1)).contiguous().view(batch_size,-1)
        try:
            invariant_middle = torch.cat([invariant_feature_two, self.cls_token, invariant_feature, self.cls_token,
                                          self.specific_a,  self.specific_v, self.specific_t],dim=1)  #invariant_middle: torch.Size([64, 10, e_dim])
            invariant_middle = invariant_middle + self.pos_embed
        except:
            invariant_middle = torch.cat([invariant_feature_two, invariant_feature,
                                      self.specific_a,  self.specific_v, self.specific_t],dim=1)

        m = self.classific_m(invariant_middle.permute(0,2,1)).contiguous().view(batch_size,-1)
        ret = {
            'M': m,
            'A': a,
            'V': v,
            'T': t,
            'fea_m': invariant_feature,  #torch.Size([B, 2, e_dim])
            'fea_m_2': invariant_feature_two,  # torch.Size([B, 2, e_dim])
            'fea_a': self.specific_a,    #torch.Size([B, 2, e_dim])
            'fea_v': self.specific_v,    #torch.Size([B, 2, e_dim])
            'fea_t': self.specific_t,    #torch.Size([B, 2, e_dim])
            'rebu_a' : rebuild_a,        #torch.Size([B, 2, e_dim])
            'rebu_v' : rebuild_v,
            'rebu_t' : rebuild_t,
            'ori_a': aud_ori,
            'ori_v': vis_ori,
            'ori_t': tex_ori,
        }
        return ret





from einops.layers.torch import Rearrange

class MLP_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MLP_Communicator(nn.Module):
    def __init__(self, token, channel, hidden_size, depth=1):
        super(MLP_Communicator, self).__init__()
        self.depth = depth
        self.token_mixer = nn.Sequential(
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel, hidden_size=hidden_size),
            Rearrange('b n d -> b d n')
        )
        self.channel_mixer = nn.Sequential(
            MLP_block(input_size=token, hidden_size=hidden_size)
        )

    def forward(self, x):
        for _ in range(self.depth):
            x = x + self.token_mixer(x)
            x = x + self.channel_mixer(x)
        return x




class BertTextEncoder(nn.Module):
    def __init__(self, config, language='cn', use_finetune=False):
        """
        language: en / cn
        """
        super(BertTextEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        # directory is fine
        if language == 'en':
            bert_path = config.bert_path_en
            self.tokenizer = tokenizer_class.from_pretrained(bert_path, do_lower_case=True)
            self.model = model_class.from_pretrained(bert_path)
        elif language == 'cn':
            bert_path = config.bert_path_cn
            self.tokenizer = tokenizer_class.from_pretrained(bert_path)
            self.model = model_class.from_pretrained(bert_path)

        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:, 0, :].long(), text[:, 1, :].float(), text[:, 2, :].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():

                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states