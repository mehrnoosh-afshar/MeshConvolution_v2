import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import SplineConv, ChebConv, GMMConv, GATConv, FeaStConv, BatchNorm


def get_edge_index_lst_from_neighbor_id_lstlst(neighbor_id_lstlst, neighbor_num_lst):
    edge_index_lst = set({})
    for i in range(len(neighbor_id_lstlst)):
        neighborhood = neighbor_id_lstlst[i]
        neighbor_num = int(neighbor_num_lst[i].item())
        for j in range(neighbor_num):
            neighbor_id = neighborhood[j]
            if (i < neighbor_id):
                edge_index_lst.add((i, neighbor_id))
            elif (i > neighbor_id):
                edge_index_lst.add((neighbor_id, i))

    edge_index_lst = torch.LongTensor(list(edge_index_lst)).cuda()
    edge_index_lst = edge_index_lst.transpose(0, 1)
    return edge_index_lst

class GVAE(nn.Module):
    def __init__(self, param, test_mode = False):
        super(GVAE,self).__init__()
        ##### Parametrs
        self.point_num = param.point_num

        self.test_mode = test_mode
        self.conv_method_list = param.conv_method_list
        # Defines the output channel of each layer
        self.channel_lst = param.channel_lst

        self.residual_rate_lst = param.residual_rate_lst

        self.type_layer_lst = param.type_layer_lst

        self.weight_num_lst = param.weight_num_lst

        self.connection_layer_fn_lst = param.connection_layer_fn_lst

        self.initial_connection_fn = param.initial_connection_fn

        self.use_vanilla_pool = param.use_vanilla_pool

        ##### I want to explore various activation functions
        self.activation = nn.ELU()

        self.batch = param.batch

        #####For Laplace computation######
        self.initial_neighbor_id_lstlst = torch.LongTensor(
            param.neighbor_id_lstlst).cuda()  # point_num*max_neighbor_num
        self.initial_neighbor_num_lst = torch.FloatTensor(param.neighbor_num_lst).cuda()  # point_num
        self.initial_max_neighbor_num = self.initial_neighbor_id_lstlst.shape[1]

        self.layer_num = len(self.channel_lst)
        ##### Layers in GAE part
        in_channel = 3 ## Since we start from positional data and working with 3 feature for each Node
        self.layer_lst =[]
        for L in range (len(self.type_layer_lst)):
            layer_type = self.type_layer_lst[L]
            conv_method = self.conv_method_list[L]

            out_channel = self.channel_lst[L]
            weight_num = self.weight_num_lst[L]
            residual_rate = self.residual_rate_lst[L]
            conv_method = self.conv_method_list[L]

            connection_info = np.load(self.connection_layer_fn_lst[l])
            print("##Layer", self.connection_layer_fn_lst[l])
            out_point_num = connection_info.shape[0]

            neighbor_num_lst = torch.FloatTensor(connection_info[:, 0].astype(float)).cuda()  # out_point_num*1
            neighbor_id_dist_lstlst = connection_info[:, 1:]  # out_point_num*(max_neighbor_num*2)
            neighbor_id_lstlst = neighbor_id_dist_lstlst.reshape((out_point_num, -1, 2))[:, :, 0]  # out_point_num*max_neighbor_num
            # neighbor_id_lstlst = torch.LongTensor(neighbor_id_lstlst)
            avg_neighbor_num = neighbor_num_lst.mean().item()
            max_neighbor_num = neighbor_id_lstlst.shape[1]

            pc_mask = torch.ones(in_point_num + 1).cuda()
            pc_mask[in_point_num] = 0
            neighbor_mask_lst = pc_mask[ torch.LongTensor(neighbor_id_lstlst)].contiguous()  # out_pn*max_neighbor_num neighbor is 1 otherwise 0
            conv_layer = ""
            if layer_type == "c":
                edge_index_lst = self.get_edge_index_lst_from_neighbor_id_lstlst(neighbor_id_lstlst, neighbor_num_lst)
                print("edge_index_lst", edge_index_lst.shape)

                if (conv_method == "Cheb"):
                    conv_layer = edge_index_lst, ChebConv(in_channel, out_channel, K=2, normalization='sym', bias=True)
                    #for p in conv_layer[-1].parameters():
                    #    p.data = p.data.cuda()
                    ii = 0
                    for p in conv_layer[-1].parameters():
                        p.data = p.data.cuda()
                        self.register_parameter("Cheb" + str(l) + "_" + str(ii), p)
                        ii += 1
                elif (conv_method == "GAT"):
                    # conv_layer = edge_index_lst,GATConv(in_channel, out_channel, heads=1)
                    if ((l != ((self.layer_num / 2) - 2)) and (l != (self.layer_num - 1))):  # not middle or last layer
                        conv_layer = edge_index_lst, GATConv(in_channel, out_channel, heads=2, concat=True)
                        out_channel = out_channel * 8
                    else:
                        conv_layer = edge_index_lst, GATConv(in_channel, out_channel, heads=2, concat=False)
                    #for p in conv_layer[-1].parameters():
                    #    p.data = p.data.cuda()
                    ii = 0
                    for p in conv_layer[-1].parameters():
                        p.data = p.data.cuda()
                        self.register_parameter("GAT" + str(l) + "_" + str(ii), p)
                        ii += 1
                elif (conv_method == "GMM"):
                     # pseudo_coordinates = self.get_GMM_pseudo_coordinates(edge_index_lst, neighbor_num_lst)  # edge_num*2
                    peint("empty")
                    # print ("pseudo_coordinates", pseudo_coordinates.shape)
                    #conv_layer = edge_index_lst, pseudo_coordinates, GMMConv(in_channel, out_channel, dim=2, kernel_size=25)
                    #ii = 0
                    #for p in conv_layer[-1].parameters():
                    #   p.data = p.data.cuda()
                    #   self.register_parameter("GMM" + str(l) + "_" + str(ii), p)
                    #   ii += 1

                elif (conv_method == "FeaST"):

                    conv_layer = edge_index_lst, FeaStConv(in_channel, out_channel, heads=32)
                    ii = 0
                    for p in conv_layer[-1].parameters():
                        p.data = p.data.cuda()
                        self.register_parameter("FeaST" + str(l) + "_" + str(ii), p)
                        ii += 1
                elif (conv_method == "vw"):
                    weights = torch.randn(out_point_num, max_neighbor_num, out_channel, in_channel) / (
                                avg_neighbor_num * weight_num)

                    weights = nn.Parameter(weights.cuda())

                    self.register_parameter("weights" + str(l), weights)

                    bias = nn.Parameter(torch.zeros(out_channel).cuda())
                    if (self.perpoint_bias == 1):
                        bias = nn.Parameter(torch.zeros(out_point_num, out_channel).cuda())
                    self.register_parameter("bias" + str(l), bias)

                    conv_layer = (weights, bias)
                elif (conv_method == "full"):

                    weights = torch.randn(weight_num, out_channel * in_channel).cuda()

                    weights = nn.Parameter(weights).cuda()

                    self.register_parameter("weights" + str(l), weights)

                    bias = nn.Parameter(torch.zeros(out_channel).cuda())

                    if (self.perpoint_bias == 1):
                        bias = nn.Parameter(torch.zeros(out_point_num, out_channel).cuda())
                    self.register_parameter("bias" + str(l), bias)

                    w_weights = torch.randn(out_point_num, max_neighbor_num, weight_num) / (
                                avg_neighbor_num * weight_num)

                    w_weights = nn.Parameter(w_weights.cuda())
                    self.register_parameter("w_weights" + str(l), w_weights)

                    conv_layer = (weights, bias, w_weights)

                #####put everythin together ########
                layer = (
                    in_channel, out_channel, in_point_num, out_point_num, weight_num, max_neighbor_num,
                    neighbor_num_lst,neighbor_id_lstlst, conv_layer, neighbor_mask_lst,zeros_batch_outpn_outchannel, conv_method, layer_type)
                in_channel = out_channel
                self.layer_lst += [layer]

            elif  layer_type == "p":
             # Pooling layers on graph signals
              pool_layer = avg_pool_neighbor_x()
             #####put everythin together ########
              layer = (
                in_channel, out_channel, in_point_num, out_point_num, weight_num, max_neighbor_num,
                neighbor_num_lst, neighbor_id_lstlst, pool_layer, neighbor_mask_lst, zeros_batch_outpn_outchannel,
                conv_method, layer_type)
              in_channel = out_channel
              self.layer_lst += [layer]
            elif layer_type == "n":
             # Normalozation layer
              Norm_layer = BatchNorm(in_channel)
              layer = (
                in_channel, out_channel, in_point_num, out_point_num, weight_num, max_neighbor_num,
                neighbor_num_lst, neighbor_id_lstlst, Norm_layer, neighbor_mask_lst, zeros_batch_outpn_outchannel,
                conv_method, layer_type)
              in_channel = out_channel
              self.layer_lst += [layer]

    def forward_one_layer_perBatch(self,x_batch,layer_info,is_final_layer = False):
        batch=x_batch.shape[0]
        in_channel, out_channel, in_pn, out_pn, weight_num,  max_neighbor_num, neighbor_num_lst,neighbor_id_lstlst, layer_property, residual_layer, residual_rate,\
        neighbor_mask_lst, zeros_batch_outpn_outchannel, conv_method ,layer_type =layer_info

        x_pad = torch.cat((x_batch, torch.zeros(batch, 1, in_channel).cuda()), 1)  # batch*(in_pn+1)*in_channel

        in_neighbors = x_pad[:, neighbor_id_lstlst]  # batch*out_pn*max_neighbor_num*in_channel

        ####compute output of convolution layer####
        out_x_conv = zeros_batch_outpn_outchannel.clone()
        if layer_type == "c":
            ####compute output of convolution layer####
            out_pc_conv = torch.FloatTensor([]).cuda()
            if ((conv_method == "GAT") or (conv_method == "Cheb") or (conv_method == "FeaST")):
                edge_index_lst, conv = layer_property  # weight_num*(out_channel*in_channel)   out_point_num* max_neighbor_num* weight_num
                for b in range(batch):
                   out_x_one = conv(x_batch[b], edge_index_lst)
                   out_x_conv = torch.cat((out_pc_conv, out_pc_one.unsqueeze(0)), 0)
            elif ((conv_method == "GMM")):
                edge_index_lst, pseudo_coordinates, conv = layer_property
                for b in range(batch):
                    out_x_one = conv(in_pc[b], edge_index_lst, pseudo_coordinates)
                    out_x_conv = torch.cat((out_pc_conv, out_pc_one.unsqueeze(0)), 0)
            elif (conv_method == "vw"):
                weights, bias = layer_property
                out_neighbors = torch.einsum('pmoi,bpmi->bpmo',
                                             [weights, in_neighbors])  # batch*out_pn*max_neighbor_num*out_channel
                out_neighbors = out_neighbors * neighbor_mask_lst.view(1, out_pn, max_neighbor_num, 1).repeat(batch, 1,1,out_channel)
                out_x_conv = out_neighbors.sum(2)
                out_x_conv = out_pc_conv + bias

            elif (conv_method == "full"):

                (weights, bias,
                raw_w_weights) = conv_layer  # weight_num*(out_channel*in_channel)   out_point_num* max_neighbor_num* weight_num
                w_weights = raw_w_weights * neighbor_mask_lst.view(out_pn, max_neighbor_num, 1).repeat(1, 1, weight_num)  # out_pn*max_neighbor_num*weight_num

                weights = torch.einsum('pmw,wc->pmc', [w_weights, weights])  # out_pn*max_neighbor_num*(out_channel*in_channel)
                weights = weights.view(out_pn, max_neighbor_num, out_channel, in_channel)

                out_neighbors = torch.einsum('pmoi,bpmi->bpmo', [weights, in_neighbors])  # batch*out_pn*max_neighbor_num*out_channel

                out_x_conv = out_neighbors.sum(2)

                out_x_conv = out_pc_conv + bias

            #### Apply activation functions ####
            if (is_final_layer == False):
                out_pc_conv = self.activation(out_pc_conv)

        elif layer_type == "n":
            out_x_conv = layer_property(x_batch)
        elif layer_type == "p":
            out_x_conv = layer_property(x_batch)

        return out_x_conv

    def forward_till_last_layer(self,x_batch):
        out_x = x_batch.clone()
        for i in range(self.layer_num):
            if (i < (self.layer_num - 1)):
                out_x = self.forward_one_layer_perBatch(out_x, self.layer_lst[i])
            else:
                out_x = self.forward_one_layer_perBatch(out_x, self.layer_lst[i], is_final_layer=True)
        return out_x

    def compute_geometric_loss_l1(self, gt_x, predict_x, weights=[]):

            if (len(weights) == 0):
                loss = torch.abs(gt_x - predict_x).mean()

                return loss

            else:
                batch = gt_x.shape[0]
                point_num = gt_x.shape[1]
                x_weights = weights.view(batch, point_num, 1).repeat(1, 1, 3)

                loss = torch.abs(gt_x * x_weights - predict_x * x_weights).sum() / (batch * 3)

                return loss

    def compute_geometric_loss_l2_MSE(self, gt_x, predict_x):
        MSE_loss = nn.MSELoss()
        return MSE_loss(gt_x,predict_x)

    def compute_laplace_loss_l1_l2(self, gt_pc_raw, predict_pc_raw):
            gt_pc = gt_pc_raw * 1
            predict_pc = predict_pc_raw * 1

            batch = gt_pc.shape[0]

            gt_pc = torch.cat((gt_pc, torch.zeros(batch, 1, 3).cuda()), 1)
            predict_pc = torch.cat((predict_pc, torch.zeros(batch, 1, 3).cuda()), 1)

            batch = gt_pc.shape[0]

            gt_pc_laplace = gt_pc[:,
                            self.initial_neighbor_id_lstlst[:, 0]]  ## batch*point_num*3 the first point is itself
            gt_pc_laplace = gt_pc_laplace * (
                        self.initial_neighbor_num_lst.view(1, self.point_num, 1).repeat(batch, 1, 3) - 1)

            for n in range(1, self.initial_max_neighbor_num):
                # print (neighbor_id_lstlst[:,n])
                neighbor = gt_pc[:, self.initial_neighbor_id_lstlst[:, n]]
                gt_pc_laplace -= neighbor

            predict_pc_laplace = predict_pc[:,
                                 self.initial_neighbor_id_lstlst[:, 0]]  ## batch*point_num*3 the first point is itself
            predict_pc_laplace = predict_pc_laplace * (
                        self.initial_neighbor_num_lst.view(1, self.point_num, 1).repeat(batch, 1, 3) - 1)

            for n in range(1, self.initial_max_neighbor_num):
                # print (neighbor_id_lstlst[:,n])
                neighbor = predict_pc[:, self.initial_neighbor_id_lstlst[:, n]]
                predict_pc_laplace -= neighbor

            loss_l1 = torch.abs(gt_pc_laplace - predict_pc_laplace).mean()

            gt_pc_curv= gt_pc_laplace.pow(2).sum(2).pow(0.5)
            predict_pc_curv = predict_pc_laplace.pow(2).sum(2).pow(0.5)
            loss_curv = (gt_pc_curv-predict_pc_curv).pow(2).mean()

            return loss_l1   , loss_curv

    def compute_laplace_Mean_Euclidean_Error(self, gt_pc_raw, predict_pc_raw):
            gt_pc = gt_pc_raw * 1
            predict_pc = predict_pc_raw * 1

            batch = gt_pc.shape[0]

            gt_pc = torch.cat((gt_pc, torch.zeros(batch, 1, 3).cuda()), 1)
            predict_pc = torch.cat((predict_pc, torch.zeros(batch, 1, 3).cuda()), 1)

            batch = gt_pc.shape[0]

            gt_pc_laplace = gt_pc[:,
                            self.initial_neighbor_id_lstlst[:, 0]]  ## batch*point_num*3 the first point is itself
            gt_pc_laplace = gt_pc_laplace * (
                        self.initial_neighbor_num_lst.view(1, self.point_num, 1).repeat(batch, 1, 3) - 1)

            for n in range(1, self.initial_max_neighbor_num):
                # print (neighbor_id_lstlst[:,n])
                neighbor = gt_pc[:, self.initial_neighbor_id_lstlst[:, n]]
                gt_pc_laplace -= neighbor

            predict_pc_laplace = predict_pc[:,
                                 self.initial_neighbor_id_lstlst[:, 0]]  ## batch*point_num*3 the first point is itself
            predict_pc_laplace = predict_pc_laplace * (
                        self.initial_neighbor_num_lst.view(1, self.point_num, 1).repeat(batch, 1, 3) - 1)

            for n in range(1, self.initial_max_neighbor_num):
                # print (neighbor_id_lstlst[:,n])
                neighbor = predict_pc[:, self.initial_neighbor_id_lstlst[:, n]]
                predict_pc_laplace -= neighbor

            error = torch.pow(torch.pow(gt_pc_laplace - predict_pc_laplace, 2).sum(2), 0.5).mean()

            # gt_pc_curv= gt_pc_laplace.pow(2).sum(2).pow(0.5)
            # predict_pc_curv = predict_pc_laplace.pow(2).sum(2).pow(0.5)
            # loss_curv = (gt_pc_curv-predict_pc_curv).pow(2).mean()

            return error  # , loss_cur

    def compute_geometric_mean_euclidean_dist_error(self, gt_pc, predict_pc, weights=[]):

            if (len(weights) == 0):
                error = (gt_pc - predict_pc).pow(2).sum(2).pow(0.5).mean()

                return error

            else:
                batch = gt_pc.shape[0]
                point_num = gt_pc.shape[1]
                channel = gt_pc.shape[2]

                dists = (gt_pc - predict_pc).pow(2).sum(2).pow(0.5) * weights
                error = dists.sum()

                return error


