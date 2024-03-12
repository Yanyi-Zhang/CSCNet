import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.word_embedding import load_word_embeddings
from model.common import MLP
    
class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class CSCNet(nn.Module):
    def __init__(self, dset, args):
        super(CSCNet, self).__init__()
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs).to(args.device)
            objs = torch.LongTensor(objs).to(args.device)
            pairs = torch.LongTensor(pairs).to(args.device)
            return attrs, objs, pairs

        # Validation - Use all pairs to validate
        self.val_attrs, self.val_objs, self.val_pairs = get_all_ids(self.dset.pairs)
        # All attrs and objs without repetition
        self.uniq_attrs, self.uniq_objs = torch.arange(len(self.dset.attrs)).long().to(args.device), \
                                          torch.arange(len(self.dset.objs)).long().to(args.device)
        if args.train_only:
            self.train_attrs, self.train_objs, self.train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            self.train_attrs, self.train_objs, self.train_pairs = self.val_attrs, self.val_objs, self.val_pairs
        
        self.attr_num = self.uniq_attrs.shape[0]
        self.obj_num = self.uniq_objs.shape[0]
        self.pair_num = self.train_pairs.shape[0]

        attr_word_emb_file = '{}_{}_attr.save'.format(args.dataset, args.emb_type)
        attr_word_emb_file = os.path.join(args.main_root, 'word embedding', attr_word_emb_file)
        obj_word_emb_file = '{}_{}_obj.save'.format(args.dataset, args.emb_type)
        obj_word_emb_file = os.path.join(args.main_root, 'word embedding', obj_word_emb_file)

        print('  Load attribute word embeddings--')
        if os.path.exists(attr_word_emb_file):
            pretrained_weight_attr = torch.load(attr_word_emb_file, map_location=args.device)
        else:
            pretrained_weight_attr = load_word_embeddings(dset.attrs, args)
            print('  Save attr word embeddings using {}'.format(args.emb_type))
            torch.save(pretrained_weight_attr, attr_word_emb_file)
        emb_dim = pretrained_weight_attr.shape[1]
        self.attr_embedder = nn.Embedding(len(dset.attrs), emb_dim).to(args.device)
        self.attr_embedder.weight.data.copy_(pretrained_weight_attr)

        print('  Load object word embeddings--')
        if os.path.exists(obj_word_emb_file):
            pretrained_weight_obj = torch.load(obj_word_emb_file, map_location=args.device)
        else:
            pretrained_weight_obj = load_word_embeddings(dset.objs, args)
            print('  Save obj word embeddings using {}'.format(args.emb_type))
            torch.save(pretrained_weight_obj, obj_word_emb_file)
        self.obj_embedder = nn.Embedding(len(dset.objs), emb_dim).to(args.device)
        self.obj_embedder.weight.data.copy_(pretrained_weight_obj)

        self.image_embedder_attr_2 = MLP(dset.feat_dim, emb_dim, num_layers=args.nlayers, relu=False, bias=True,
                                       dropout=True, norm=True, layers=[])
        self.image_embedder_obj_1 = MLP(dset.feat_dim, emb_dim, num_layers=args.nlayers, relu=False, bias=True,
                                      dropout=True, norm=True, layers=[])
        self.image_embedder_both = MLP(dset.feat_dim, emb_dim, num_layers=args.nlayers, relu=False, bias=True,
                                       dropout=True, norm=True, layers=[])
        self.score_attr = RelationNetwork(emb_dim*2,400)
        self.score_obj = RelationNetwork(emb_dim*2,400)

        self.pred_attr_con = RelationNetwork(emb_dim*2, 400)
        self.pred_obj_con = RelationNetwork(emb_dim*2, 400)
        self.emb_dim = emb_dim
        # static inputs
        if not args.update_word_features:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False
        
        self.projection = MLP(emb_dim*2, emb_dim, relu=True, bias=True, dropout=True, norm=True,
                              num_layers=2, layers=[])

        self.img_obj_compose = MLP(emb_dim+dset.feat_dim, emb_dim, relu=True, bias=True, dropout=True, norm=True,
                                   num_layers=2, layers=[emb_dim])
        self.img_attr_compose = MLP(emb_dim+dset.feat_dim, emb_dim, relu=True, bias=True, dropout=True, norm=True,
                                   num_layers=2, layers=[emb_dim])
        
        self.alpha = args.alpha 
        self.τ = args.cosine_scale 
                
    def compose(self, attrs, objs):
        attrs, objs = self.attr_embedder(attrs), self.obj_embedder(objs)
        inputs = torch.cat([attrs, objs], -1)
        output = self.projection(inputs)
        return output

    def val_forward(self, input_batch):
        x = input_batch[0]
        BATCH_SIZE = x.shape[0]
        del input_batch

        # Map the input image embedding
        ω_a_x_2 = self.image_embedder_attr_2(x)
        ω_c_x = self.image_embedder_both(x)
        ω_o_x_1 = self.image_embedder_obj_1(x)

        # Acquire word embeddings of all attrs and objs
        v_a = self.attr_embedder(self.uniq_attrs)
        v_o = self.obj_embedder(self.uniq_objs)

        # Pred composition
        ω_o_x_pre = ω_o_x_1.unsqueeze(0).repeat(self.obj_num,1,1)
        ω_o_x_pre = torch.transpose(ω_o_x_pre,0,1)
        v_o_in = v_o.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        score_obj_ori = torch.cat((ω_o_x_pre,v_o_in),2).view(-1, self.emb_dim*2)
        score_obj_ori = self.score_obj(score_obj_ori).view(-1,self.obj_num)
        o_star = torch.argmax(score_obj_ori, dim=-1)
        v_o_star = self.obj_embedder(o_star)

        # Pred obj
        β = self.img_obj_compose(torch.cat((x, v_o_star), dim=-1))
        β = β.unsqueeze(0).repeat(self.attr_num,1,1)
        β = torch.transpose(β,0,1)
        v_a_in = v_a.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        score_attr_con = torch.cat((β, v_a_in),2).view(-1, self.emb_dim*2)
        score_attr_con = self.pred_attr_con(score_attr_con).view(-1,self.attr_num)

        # Pred attr
        ω_a_x_pre = ω_a_x_2.unsqueeze(0).repeat(self.attr_num,1,1)
        ω_a_x_pre = torch.transpose(ω_a_x_pre,0,1)
        v_a_in = v_a.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        score_attr_ori = torch.cat((ω_a_x_pre,v_a_in),2).view(-1, self.emb_dim*2)
        score_attr_ori = self.score_attr(score_attr_ori).view(-1,self.attr_num)
        a_star = torch.argmax(score_attr_ori, dim=-1)
        v_a_star = self.attr_embedder(a_star)

        # Pred obj
        β = self.img_attr_compose(torch.cat((x, v_a_star), dim=-1))
        β = β.unsqueeze(0).repeat(self.obj_num,1,1)
        β = torch.transpose(β,0,1)
        v_o_in = v_o.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        score_obj_con = torch.cat((β, v_o_in),2).view(-1, self.emb_dim*2)
        score_obj_con = self.pred_obj_con(score_obj_con).view(-1,self.obj_num)

        # Pred composition
        v_ao = self.compose(self.val_attrs, self.val_objs)
        ω_c_x = F.normalize(ω_c_x, dim=-1)
        v_ao = F.normalize(v_ao, dim=-1)
        score_pair = ω_c_x @ v_ao.t()

        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = (1-self.alpha) * score_pair[:, self.dset.all_pair2idx[pair]]
            scores[pair] += self.alpha * (score_attr_ori[:, self.dset.attr2idx[pair[0]]] * score_obj_con[:, self.dset.obj2idx[pair[1]]])
            scores[pair] += self.alpha * (score_attr_con[:, self.dset.attr2idx[pair[0]]] * score_obj_ori[:, self.dset.obj2idx[pair[1]]])
        return None, scores

    def train_forward(self, input_batch):
        x, a, o, c = input_batch[0], input_batch[1], input_batch[2], input_batch[3]
        BATCH_SIZE = x.shape[0]
        del input_batch

        # Map the input image embedding
        ω_a_x_2 = self.image_embedder_attr_2(x)
        ω_c_x = self.image_embedder_both(x)
        ω_o_x_1 = self.image_embedder_obj_1(x)

        # Acquire word embeddings of all attrs and objs
        v_a = self.attr_embedder(self.uniq_attrs)
        v_o = self.obj_embedder(self.uniq_objs)

        # Pred obj
        ω_o_x_pre = ω_o_x_1.unsqueeze(0).repeat(self.obj_num,1,1)
        ω_o_x_pre = torch.transpose(ω_o_x_pre,0,1)
        v_o_in = v_o.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        score_obj_ori = torch.cat((ω_o_x_pre,v_o_in),2).view(-1, self.emb_dim*2)
        score_obj_ori = self.score_obj(score_obj_ori).view(-1,self.obj_num)
        score_obj_ori = score_obj_ori / self.τ
        
        o_star = torch.argmax(score_obj_ori, dim=-1)
        v_o_star = self.obj_embedder(o_star)

        # Pred attr
        β = self.img_obj_compose(torch.cat((x, v_o_star), dim=-1))
        β = β.unsqueeze(0).repeat(self.attr_num,1,1)
        β = torch.transpose(β,0,1)
        v_a_in = v_a.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        score_attr_con = torch.cat((β, v_a_in),2).view(-1, self.emb_dim*2)
        score_attr_con = self.pred_attr_con(score_attr_con).view(-1,self.attr_num)
        score_attr_con = score_attr_con / self.τ

        # Pred attr
        ω_a_x_pre = ω_a_x_2.unsqueeze(0).repeat(self.attr_num,1,1)
        ω_a_x_pre = torch.transpose(ω_a_x_pre,0,1)
        v_a_in = v_a.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        score_attr_ori = torch.cat((ω_a_x_pre,v_a_in),2).view(-1, self.emb_dim*2)
        score_attr_ori = self.score_attr(score_attr_ori).view(-1,self.attr_num)
        score_attr_ori = score_attr_ori / self.τ
        a_star = torch.argmax(score_attr_ori, dim=-1)
        v_a_star = self.attr_embedder(a_star)

        # Pred obj
        β = self.img_attr_compose(torch.cat((x, v_a_star), dim=-1))
        β = β.unsqueeze(0).repeat(self.obj_num,1,1)
        β = torch.transpose(β,0,1)
        v_o_in = v_o.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        score_obj_con = torch.cat((β, v_o_in),2).view(-1, self.emb_dim*2)
        score_obj_con = self.pred_obj_con(score_obj_con).view(-1,self.obj_num)
        score_obj_con = score_obj_con / self.τ

        # Pred composition
        v_ao = self.compose(self.train_attrs, self.train_objs)
        ω_c_x = F.normalize(ω_c_x, dim=-1)
        v_ao = F.normalize(v_ao, dim=-1)
        score_pair = ω_c_x @ v_ao.t()
        score_pair = score_pair / self.τ

        L_o_c = F.cross_entropy(score_obj_ori, o)
        L_a_c = F.cross_entropy(score_attr_ori, a)
        L_ao = F.cross_entropy(score_pair, c) # Eq.11
        L_a_after = F.cross_entropy(score_obj_con, o) # Eq.10
        L_o_after = F.cross_entropy(score_attr_con, a)

        return (L_a_c + L_o_c + L_a_after + L_o_after) / 4 + L_ao, None # Eq.12

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
            return loss, pred
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
            return loss, pred