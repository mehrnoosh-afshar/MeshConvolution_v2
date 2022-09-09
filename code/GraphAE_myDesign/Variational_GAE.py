
import pdb
import numpy as np
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SplineConv, ChebConv, GMMConv, GATConv, FeaStConv, BatchNorm, \
    GATv2Conv, ARMAConv, \
    TransformerConv

class Pooling_Layer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, in_channels: int, out_channels: int, layer_dim_info):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        out_point_num , in_point_num , max_neighbor_num, avg_neighbor_num = layer_dim_info
        self.in_point_num = in_point_num
        self.out_point_num = out_point_num
        self.max_neighbor_num=max_neighbor_num
        self.avg_neighbor_num = avg_neighbor_num
        p_neighbors = ""
        weight_res = ""

        if (out_point_num != in_point_num):
            p_neighbors = nn.Parameter((torch.randn(out_point_num, max_neighbor_num) / (avg_neighbor_num)).cuda())
            self.register_parameter("p_neighbors" , p_neighbors)

        if (self.out_channels != self.in_channels):
            weight_res = torch.randn(self.out_channels, self.in_channels)
            # self.normalize_weights(weight_res)
            weight_res = weight_res / self.out_channels
            weight_res = nn.Parameter(weight_res.cuda())
            self.register_parameter("weight_res" , weight_res)


    def forward(self, in_pc_pad, neighbor_id_lstlst,neighbor_mask_lst):
        zeros_batch_outpn_outchannel = torch.zeros((in_pc_pad.shape[0], self.out_point_num, self.out_channels)).cuda()
        out_pc = zeros_batch_outpn_outchannel.clone()
        if (self.in_channels != self.out_channels):
            in_pc_pad = torch.einsum('oi,bpi->bpo', [self.weight_res, in_pc_pad])

        out_pc = []
        if (self.in_point_num == self.out_point_num):
            out_pc = in_pc_pad[:, 0:self.in_point_num].clone()
        else:
           in_neighbors = in_pc_pad[:, neighbor_id_lstlst]  # batch*out_pn*max_neighbor_num*out_channel
           p_neighbors = torch.abs(self.p_neighbors) * neighbor_mask_lst
           p_neighbors_sum = p_neighbors.sum(1) + 1e-8  # out_pn
           p_neighbors = p_neighbors / p_neighbors_sum.view(self.out_point_num, 1).repeat(1, self.max_neighbor_num)

           # print("in pooling layer","p_neighbors",p_neighbors.shape,"in_neighbors",in_neighbors.shape)    

           out_pc = torch.einsum('pm,bpmo->bpo', [p_neighbors, in_neighbors])
           
        return  out_pc

class full_conv_Layer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, in_channels: int, out_channels: int, layer_dim_info , weight_num = 9, perpoint_bias=0):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        out_point_num, in_point_num, max_neighbor_num, avg_neighbor_num = layer_dim_info
        self.in_point_num = in_point_num
        self.out_point_num = out_point_num
        self.max_neighbor_num = max_neighbor_num
        self.avg_neighbor_num = avg_neighbor_num
        self.perpoint_bias = perpoint_bias
        self.weight_num=weight_num
        weights = torch.randn(weight_num, self.out_channels * self.in_channels).cuda()
        weights = nn.Parameter(weights).cuda()
        self.register_parameter("weights" , weights)

        bias = nn.Parameter(torch.zeros(self.out_channels ).cuda())
        if (self.perpoint_bias == 1):
            bias = nn.Parameter(torch.zeros(out_point_num, self.out_channels).cuda())
        self.register_parameter("bias", bias)

        w_weights = torch.randn(out_point_num, max_neighbor_num, weight_num) / (avg_neighbor_num * weight_num)

        w_weights = nn.Parameter(w_weights.cuda())
        self.register_parameter("w_weights" , w_weights)


    def forward(self, in_pc_pad, neighbor_id_lstlst, neighbor_mask_lst):
        zeros_batch_outpn_outchannel = torch.zeros((in_pc_pad.shape[0], self.out_point_num, self.out_channels)).cuda()
        out_pc = zeros_batch_outpn_outchannel.clone()
        in_neighbors = in_pc_pad[:, neighbor_id_lstlst]  # batch*out_pn*max_neighbor_num*in_channel

            
        w_weights2 = self.w_weights*neighbor_mask_lst.view(self.out_point_num, self.max_neighbor_num, 1).repeat(1,1,self.weight_num) #out_pn*max_neighbor_num*weight_num

        
        weights = torch.einsum('pmw,wc->pmc',[w_weights2, self.weights]) #out_pn*max_neighbor_num*(out_channel*in_channel)
        weights = weights.view(self.out_point_num, self.max_neighbor_num, self.out_channels,self.in_channels)
        out_neighbors = torch.einsum('pmoi,bpmi->bpmo',[weights, in_neighbors]) #batch*out_pn*max_neighbor_num*out_channel
            
        out_pc = out_neighbors.sum(2)
            
        out_pc = out_pc + self.bias
            

        return out_pc


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


def get_GMM_pseudo_coordinates(edge_index_lst, neighbor_num_lst):
    pseudo_coordinates = neighbor_num_lst[edge_index_lst.transpose(0, 1)]
    pseudo_coordinates = 1 / torch.sqrt(pseudo_coordinates)

    return pseudo_coordinates


class Encoder(nn.Module):
    def __init__(self, param, test_mode=False):
        super(Encoder, self).__init__()

        ##### Parametrs
        self.point_num = param.point_num

        self.test_mode = test_mode
        self.conv_method_list = param.conv_method_list

        # Defines the output channel of each layer
        self.channel_lst = param.channel_lst
        self.latent_dim = param.latent_dim
        # self.residual_rate_lst = param.residual_rate_lst

        self.weight_num_lst = param.weight_num_lst

        self.connection_layer_fn_lst = param.connection_layer_fn_lst

        self.initial_connection_fn = param.initial_connection_fn

        self.use_vanilla_pool = param.use_vanilla_pool

        ##### I want to explore various activation functions
        self.activation = nn.ModuleDict({"Relu": nn.ELU(), "Tanh": nn.Tanh()})
        self.perpoint_bias = param.perpoint_bias

        #####For Laplace computation######
        self.initial_neighbor_id_lstlst = torch.LongTensor(
            param.neighbor_id_lstlst).cuda()  # point_num*max_neighbor_num
        self.initial_neighbor_num_lst = torch.FloatTensor(param.neighbor_num_lst).cuda()  # point_num
        self.initial_max_neighbor_num = self.initial_neighbor_id_lstlst.shape[1]

        self.layer_num = len(self.channel_lst)

        in_point_num = self.point_num
        in_channel = 3  ## Since we start from positional data and working with 3 feature for each Node
        self.layer_lst = []
        self.layer_info_list = []
        self.Layers = nn.ModuleList()
        self.NormLayers = nn.ModuleList()

        for L in range(len(self.channel_lst)):
        
            conv_method = self.conv_method_list[L]

            out_channel = self.channel_lst[L]
            weight_num = self.weight_num_lst[L]
            

            connection_info = np.load(self.connection_layer_fn_lst[L])

            print("##Layer", self.connection_layer_fn_lst[L])

            out_point_num = connection_info.shape[0]

            neighbor_num_lst = torch.FloatTensor(connection_info[:, 0].astype(float)).cuda()  # out_point_num*1
            neighbor_id_dist_lstlst = connection_info[:, 1:]  # out_point_num*(max_neighbor_num*2)
            neighbor_id_lstlst = neighbor_id_dist_lstlst.reshape((out_point_num, -1, 2))[:, :, 0]  # out_point_num*max_neighbor_num
            # neighbor_id_lstlst = torch.LongTensor(neighbor_id_lstlst)
            avg_neighbor_num = neighbor_num_lst.mean().item()
            max_neighbor_num = neighbor_id_lstlst.shape[1]

            pc_mask = torch.ones(in_point_num + 1).cuda()
            pc_mask[in_point_num] = 0
            neighbor_mask_lst = pc_mask[
                torch.LongTensor(neighbor_id_lstlst)].contiguous()  # out_pn*max_neighbor_num neighbor is 1 otherwise 0
            conv_layer = ""

            

            #### Put all together and build a list of layers ##########
            layer_info = (in_channel, out_channel, in_point_num, out_point_num, weight_num, max_neighbor_num,avg_neighbor_num, neighbor_num_lst,
                neighbor_id_lstlst, conv_method,
                neighbor_mask_lst, L)

            
            Dum_var = self.convlayer(layer_info)
            self.Layers.add_module(f"f{L}", Dum_var[0])
            
            out_channel = Dum_var[1]

            layer_info = (in_channel, out_channel, in_point_num, out_point_num, weight_num, max_neighbor_num,avg_neighbor_num, neighbor_num_lst,
            neighbor_id_lstlst, conv_method,
            neighbor_mask_lst, L)

            in_channel = out_channel
            in_point_num = out_point_num

            self.layer_info_list += [layer_info]

            self.NormLayers.add_module(f"n{L}",BatchNorm(out_point_num))
            self.out_channel = out_channel
        

        ### Add the final layer to split feature vector into miu and std for VAE #######

        layer_info = self.layer_info_list[-1]
        in_channel, out_channel, in_pn, out_pn, weight_num, max_neighbor_num,avg_neighbor_num, neighbor_num_lst, neighbor_id_lstlst, conv_method, \
        neighbor_mask_lst, layer_number = layer_info

        # layer_info = list(layer_info)
        # layer_info[0] = out_channel
        # layer_info[1] = self.latent_dim

        # Applied only linear FC on feature vector in a graph
        # self.Layers.add_module(f"MIU", nn.Linear(in_features=out_channel, out_features=self.latent_dim))
        # self.Layers.add_module(f"SIGMA", nn.Linear(in_features=out_channel, out_features=self.latent_dim))

        self.f_layer_dim = out_channel*out_pn
        # Final layer to get MIU and Sigma without activation and reshape
        # self.MIU_noreshape = nn.Linear(in_features=out_channel, out_features=self.latent_dim)
        # self.Sigma_noreshape= nn.Linear(in_features=out_channel, out_features=self.latent_dim)

        # Final layer to get MIU and Sigma without activation and WITH reshape
      
        self.MIU_reshape = nn.Linear(in_features=out_channel*out_pn, out_features=self.latent_dim)
        self.Sigma_reshape = nn.Linear(in_features=out_channel*out_pn, out_features=self.latent_dim)


    def get_final_layer_dim(self):
        return self.f_layer_dim

    def convlayer(self, layer_info, test_mode=False, normalization=False, activation=True):

        in_channel, out_channel, in_pn, out_pn, weight_num, max_neighbor_num,avg_neighbor_num, neighbor_num_lst, neighbor_id_lstlst, conv_method, \
        neighbor_mask_lst, layer_number = layer_info

        # edge_index_lst = get_edge_index_lst_from_neighbor_id_lstlst(neighbor_id_lstlst, neighbor_num_lst)
        # print("edge_index_lst", edge_index_lst.shape)
        L = layer_number
        # avg_neighbor_num = neighbor_num_lst.mean().item()
        # zeros_batch_outpn_outchannel = torch.zeros((self.batch, out_pn, out_channel)).cuda()

        if (conv_method == "Cheb"):
            conv_layer = ChebConv(in_channel, out_channel, K=2, normalization='sym', bias=True)
            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("Cheb" + str(L) + "_" + str(ii), p)
            #    ii += 1
        elif (conv_method == "GAT"):
            if ((L != ((self.layer_num / 2) - 2)) and (L != (self.layer_num - 1))):  # not middle or last layer
                # conv_layer = edge_index_lst, GATConv(in_channel, out_channel, heads=2, concat=True)
                conv_layer = GATv2Conv(in_channel, out_channel, heads=2, concat=True)
                out_channel = out_channel * 2
            else:
                conv_layer = GATConv(in_channel, out_channel, heads=2, concat=False)
            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("GAT" + str(L) + "_" + str(ii), p)
            #    ii += 1

        elif (conv_method == "GMM"):
            print("empty")
            # pseudo_coordinates = get_GMM_pseudo_coordinates(edge_index_lst, neighbor_num_lst)  # edge_num*2

            # print ("pseudo_coordinates", pseudo_coordinates.shape)
            # conv_layer = pseudo_coordinates, GMMConv(in_channel, out_channel, dim=2, kernel_size=25)
            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("GMM" + str(L) + "_" + str(ii), p)
            #    ii += 1

        elif (conv_method == "FeaST"):

            conv_layer = FeaStConv(in_channel, out_channel, heads=32)
            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("FeaST" + str(L) + "_" + str(ii), p)
            #    ii += 1

        elif (conv_method == "full"):
            
            layer_dim_info = (out_pn , in_pn , max_neighbor_num, avg_neighbor_num) 
            conv_layer = full_conv_Layer(in_channel,out_channel,layer_dim_info)

        elif (conv_method == "pool"):
            layer_dim_info = (out_pn , in_pn , max_neighbor_num, avg_neighbor_num) 
            conv_layer = Pooling_Layer(in_channel,out_channel,layer_dim_info)

        elif (conv_method == "ARMA"):
            conv_layer = ARMAConv(in_channel, out_channel, num_stacks=4, num_layers=2,
                                                  shared_weights=True)
            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("ARMA" + str(L) + "_" + str(ii), p)
            #    ii += 1

        elif (conv_method == "Trans"):
            if ((L != ((self.layer_num / 2) - 2)) and (L != (self.layer_num - 1))):  # not middle or last layer
                conv_layer = TransformerConv(in_channel, out_channel, heads=2)
                out_channel = out_channel * 2
            else:
                conv_layer = TransformerConv(in_channel, out_channel, heads=2, concat=False)

            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("Trans" + str(L) + "_" + str(ii), p)
            #    ii += 1

        return conv_layer, out_channel

    def forward_one_layer(self, x, layer_object,layer_info, is_final_layer=False):
        batch = x.shape[0]
        conv = layer_object


        in_channel, out_channel, in_pn, out_pn, weight_num, max_neighbor_num,avg_neighbor_num, neighbor_num_lst, neighbor_id_lstlst, conv_method, \
        neighbor_mask_lst, layer_number = layer_info



        edge_index_lst = get_edge_index_lst_from_neighbor_id_lstlst(neighbor_id_lstlst, neighbor_num_lst)
        # print("edge_index_lst", edge_index_lst.shape)
        avg_neighbor_num = neighbor_num_lst.mean().item()

        out_point_num = out_pn
        

        x_pad = torch.cat((x, torch.zeros(batch,1, in_channel).cuda()), 1)  # batch*(in_pn+1)*in_channel

        in_neighbors = x_pad[:, neighbor_id_lstlst]  # batch*out_pn*max_neighbor_num*in_channel


        ####compute output of convolution layer####
        zeros_batch_outpn_outchannel = torch.zeros((batch,out_pn,out_channel)).cuda()
        out_x_conv = zeros_batch_outpn_outchannel.clone()
        if ((conv_method == "GAT") or (conv_method == "Cheb") or (conv_method == "FeaST") or (
                    conv_method == "ARMA") or (conv_method == "Trans")):
                for b in range(batch):
                  out_x_one = conv(x[b], edge_index_lst)
                  out_x_conv[b,:,:] = out_x_one

        elif ((conv_method == "GMM")):
              edge_index_lst, pseudo_coordinates, conv = layer_property
              out_x_conv = conv(x, edge_index_lst, pseudo_coordinates)


        elif ((conv_method == "full") or (conv_method=="pool")):
              
              out_x_conv = conv(x_pad, neighbor_id_lstlst,neighbor_mask_lst)

        return out_x_conv


    def forward(self, x):
        hidden_states=[]
        i = 0
        Li = len(self.Layers)
        for (layer,normlayer) in zip(self.Layers,self.NormLayers):
            if i < Li-1:
               x = self.forward_one_layer(x,layer,self.layer_info_list[i])
               # x  = normlayer(x)
               x =  self.activation["Relu"](x)
               i += 1
            else:
                x = self.forward_one_layer(x, layer, self.layer_info_list[i])
                # out_x_conv = normlayer(out_x_conv)
                # out_x_conv = self.activation["Tanh"](out_x_conv)
                i += 1

        # If reshape is required use this line
        x_flatten = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        #x_mu = self.activation["Relu"](self.MIU_reshape(x_flatten))
        #x_var = self.activation["Relu"](self.Sigma_reshape(x_flatten)) # t=it gives the logvar log(SIGMA**2)
        batch =x.size(0)
        x_mu = torch.zeros(batch,self.latent_dim).cuda()
        x_var = torch.zeros(batch,self.latent_dim).cuda()
        for b in range(batch):
            x_mu_one = self.MIU_reshape(x_flatten[b])
            x_var_one = self.Sigma_reshape(x_flatten[b])
            x_mu[b,:] = x_mu_one
            x_var[b,:] = x_var_one



        #x_mu = self.MIU_reshape(x_flatten)
        #x_var = self.Sigma_reshape(x_flatten)
        return x_mu, x_var


class Decoder(nn.Module):

    def __init__(self, param ,in_channel,enc_final_layer_dim, test_mode=False):
        super(Decoder, self).__init__()
 

        ############  Parametrs ################
        self.point_num = int(enc_final_layer_dim/in_channel)
        self.in_channel= in_channel
        self.test_mode = test_mode
        self.conv_method_list = param.conv_method_list
        self.enc_final_layer_dim = enc_final_layer_dim



        # Defines the output channel of each layer
        self.channel_lst = param.channel_lst
        self.latent_dim = param.latent_dim
        # self.residual_rate_lst = param.residual_rate_lst

        self.weight_num_lst = param.weight_num_lst

        self.connection_layer_fn_lst = param.connection_layer_fn_lst

        self.initial_connection_fn = param.initial_connection_fn

        self.use_vanilla_pool = param.use_vanilla_pool

        ##### I want to explore various activation functions
        self.activation = nn.ModuleDict({"Relu": nn.ELU(), "Tanh": nn.Tanh()})
        self.perpoint_bias = param.perpoint_bias
        self.batch = param.batch

        #####For Laplace computation######
        self.initial_neighbor_id_lstlst = torch.LongTensor(
            param.neighbor_id_lstlst).cuda()  # point_num*max_neighbor_num
        self.initial_neighbor_num_lst = torch.FloatTensor(param.neighbor_num_lst).cuda()  # point_num
        self.initial_max_neighbor_num = self.initial_neighbor_id_lstlst.shape[1]
        self.layer_num = len(self.channel_lst)
        
        self.layer_lst = []
        self.layer_info_list = []
        self.Layers = nn.ModuleList()
        self.NormLayers = nn.ModuleList()

        
        in_point_num = self.point_num
       
        
        for L in range(len(self.channel_lst)):

            conv_method = self.conv_method_list[L]
            out_channel = self.channel_lst[L]
            weight_num = self.weight_num_lst[L]
            # residual_rate = self.residual_rate_lst[L]

            connection_info = np.load(self.connection_layer_fn_lst[L])

            print("##Layer", self.connection_layer_fn_lst[L])

            out_point_num = connection_info.shape[0]

            neighbor_num_lst = torch.FloatTensor(connection_info[:, 0].astype(float)).cuda()  # out_point_num*1
            neighbor_id_dist_lstlst = connection_info[:, 1:]  # out_point_num*(max_neighbor_num*2)
            neighbor_id_lstlst = neighbor_id_dist_lstlst.reshape((out_point_num, -1, 2))[:, :,
                                 0]  # out_point_num*max_neighbor_num
            # neighbor_id_lstlst = torch.LongTensor(neighbor_id_lstlst)
            avg_neighbor_num = neighbor_num_lst.mean().item()
            max_neighbor_num = neighbor_id_lstlst.shape[1]

            pc_mask = torch.ones(in_point_num + 1).cuda()
            pc_mask[in_point_num] = 0
            neighbor_mask_lst = pc_mask[
                torch.LongTensor(neighbor_id_lstlst)].contiguous()  # out_pn*max_neighbor_num neighbor is 1 otherwise 0
            conv_layer = ""

            layer_info = (
                in_channel, out_channel, in_point_num, out_point_num, weight_num, max_neighbor_num,avg_neighbor_num, neighbor_num_lst,
                neighbor_id_lstlst, conv_method,
                neighbor_mask_lst, L)

           
            #### Put all together and build a list of layers ##########
            Dum_var = self.convlayer(layer_info)
            self.Layers.add_module(f"f{L}", Dum_var[0])

            out_channel = Dum_var[1]
            
            layer_info = (in_channel, out_channel, in_point_num, out_point_num, weight_num, max_neighbor_num,avg_neighbor_num, neighbor_num_lst,
            neighbor_id_lstlst, conv_method,
            neighbor_mask_lst, L)

            in_channel = out_channel
            in_point_num = out_point_num

            self.layer_info_list += [layer_info]
            self.NormLayers.add_module(f"n{L}", BatchNorm(out_point_num))

        # First FC layer to bring z to general dimension
        self.fc = nn.Linear(in_features=self.latent_dim, out_features= self.enc_final_layer_dim)

    def convlayer(self, layer_info, test_mode=False, normalization=False, activation=True):

        in_channel, out_channel, in_pn, out_pn, weight_num, max_neighbor_num,avg_neighbor_num, neighbor_num_lst, neighbor_id_lstlst, conv_method, \
        neighbor_mask_lst, layer_number = layer_info

        # edge_index_lst = get_edge_index_lst_from_neighbor_id_lstlst(neighbor_id_lstlst, neighbor_num_lst)
        # print("edge_index_lst", edge_index_lst.shape)
        L = layer_number
        # avg_neighbor_num = neighbor_num_lst.mean().item()
        # zeros_batch_outpn_outchannel = torch.zeros((self.batch, out_pn, out_channel)).cuda()

        
        if (conv_method == "Cheb"):
            conv_layer = ChebConv(in_channel, out_channel, K=2, normalization='sym', bias=True)
            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("Cheb" + str(L) + "_" + str(ii), p)
            #    ii += 1
        elif (conv_method == "GAT"):
            if ((L != ((self.layer_num / 2) - 2)) and (L != (self.layer_num - 1))):  # not middle or last layer
                # conv_layer = edge_index_lst, GATConv(in_channel, out_channel, heads=2, concat=True)
                conv_layer = GATv2Conv(in_channel, out_channel, heads=2, concat=True)
                out_channel = out_channel * 2
            else:
                conv_layer = GATConv(in_channel, out_channel, heads=2, concat=False)
            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("GAT" + str(L) + "_" + str(ii), p)
            #    ii += 1

        elif (conv_method == "GMM"):
            print("empty")
            # pseudo_coordinates = get_GMM_pseudo_coordinates(edge_index_lst, neighbor_num_lst)  # edge_num*2

            # print ("pseudo_coordinates", pseudo_coordinates.shape)
            # conv_layer = pseudo_coordinates, GMMConv(in_channel, out_channel, dim=2, kernel_size=25)
            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("GMM" + str(L) + "_" + str(ii), p)
            #    ii += 1

        elif (conv_method == "FeaST"):

            conv_layer = FeaStConv(in_channel, out_channel, heads=32)
            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("FeaST" + str(L) + "_" + str(ii), p)
            #    ii += 1

        elif (conv_method == "full"):
            
            layer_dim_info = (out_pn , in_pn , max_neighbor_num, avg_neighbor_num) 
            conv_layer = full_conv_Layer(in_channel,out_channel,layer_dim_info)

        elif (conv_method == "pool"):
            layer_dim_info = (out_pn , in_pn , max_neighbor_num, avg_neighbor_num) 
            conv_layer = Pooling_Layer(in_channel,out_channel,layer_dim_info)

        elif (conv_method == "ARMA"):
            conv_layer = ARMAConv(in_channel, out_channel, num_stacks=4, num_layers=2,
                                                  shared_weights=True)
            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("ARMA" + str(L) + "_" + str(ii), p)
            #    ii += 1

        elif (conv_method == "Trans"):
            if ((L != ((self.layer_num / 2) - 2)) and (L != (self.layer_num - 1))):  # not middle or last layer
                conv_layer = TransformerConv(in_channel, out_channel, heads=2)
                out_channel = out_channel * 2
            else:
                conv_layer = TransformerConv(in_channel, out_channel, heads=2, concat=False)

            # ii = 0
            # for p in conv_layer[-1].parameters():
            #    p.data = p.data.cuda()
            #    self.register_parameter("Trans" + str(L) + "_" + str(ii), p)
            #    ii += 1
      
        return conv_layer, out_channel

    def forward_one_layer(self, x, layer_object,layer_info, is_final_layer=False):
        batch = x.shape[0]
        conv = layer_object


        in_channel, out_channel, in_pn, out_pn, weight_num, max_neighbor_num,avg_neighbor_num, neighbor_num_lst, neighbor_id_lstlst, conv_method, \
        neighbor_mask_lst, layer_number = layer_info



        edge_index_lst = get_edge_index_lst_from_neighbor_id_lstlst(neighbor_id_lstlst, neighbor_num_lst)
        # print("edge_index_lst", edge_index_lst.shape)
        avg_neighbor_num = neighbor_num_lst.mean().item()

        out_point_num = out_pn
        

        x_pad = torch.cat((x, torch.zeros(batch,1, in_channel).cuda()), 1)  # batch*(in_pn+1)*in_channel

        in_neighbors = x_pad[:, neighbor_id_lstlst]  # batch*out_pn*max_neighbor_num*in_channel


        ####compute output of convolution layer####
        zeros_batch_outpn_outchannel = torch.zeros((batch,out_pn,out_channel)).cuda()
        out_x_conv = zeros_batch_outpn_outchannel.clone()
        if ((conv_method == "GAT") or (conv_method == "Cheb") or (conv_method == "FeaST") or (
                    conv_method == "ARMA") or (conv_method == "Trans")):
                for b in range(batch):
                  out_x_one = conv(x[b], edge_index_lst)
                  out_x_conv[b,:,:] = out_x_one

        elif ((conv_method == "GMM")):
              edge_index_lst, pseudo_coordinates, conv = conv_method
              out_x_conv = conv(x, edge_index_lst, pseudo_coordinates)


        elif ((conv_method == "full") or (conv_method=="pool")):
              
              out_x_conv = conv(x_pad, neighbor_id_lstlst,neighbor_mask_lst)

       
        return out_x_conv

    def forward(self, x):

        batch = x.size(0)
        xc = torch.FloatTensor([]).cuda()
        for b in range(batch):
            x_one = self.fc(x[b])
            xc = torch.cat( (xc, x_one), 0)
        x = xc
        #x = self.fc(x)
        #x = self.activation["Relu"](x)
        # reshape it to last layer size
        x = x.view(batch, self.point_num , self.in_channel)  # unflatten batch of feature vectors to a batch of multi-channel feature maps
        

        ####compute output of convolution layer####
        i = 0
        Li = len(self.Layers)
        for (layer, normlayer) in zip(self.Layers, self.NormLayers):
            # print(layer)

            if i < Li - 1:
                x = self.forward_one_layer(x, layer, self.layer_info_list[i])
                # x = normlayer(x)
                x = self.activation["Relu"](x)
                i += 1
            else:
                x = self.forward_one_layer(x, layer, self.layer_info_list[i])
                # out_x_conv = normlayer(out_x_conv)
                # out_x_conv = self.activation["Tanh"](out_x_conv)
                i += 1

        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self,enc_param,dec_param,test_mode=False):
        super(VariationalAutoencoder, self).__init__()
        self.test_mode = test_mode
        self.encoder = Encoder(enc_param)
        self.decoder = Decoder(dec_param,self.encoder.out_channel,self.encoder.f_layer_dim)
        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        latent_mu, latent_var = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_var)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_var

    def latent_sample(self, mu, var):
        if (self.test_mode==False):
            # the reparameterization trick

            z = mu+ (0.5*var).exp()*self.N.sample(mu.shape)
            return z
        else:
            return mu

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
        return MSE_loss(gt_x, predict_x)

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

        gt_pc_curv = gt_pc_laplace.pow(2).sum(2).pow(0.5)
        predict_pc_curv = predict_pc_laplace.pow(2).sum(2).pow(0.5)
        loss_curv = (gt_pc_curv - predict_pc_curv).pow(2).mean()

        return loss_l1, loss_curv

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

    def vae_loss(self,gt_x, predict_x, mu, logvar, weights=[]):

       variational_beta = 1
       recon_loss = self.compute_geometric_loss_l1(gt_x, predict_x)

       # KL-divergence between the prior distribution over latent vectors
       # (the one we are going to sample from when generating new images)
       # and the distribution estimated by the generator for the given image.
       kldivergence_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),1)).mean()
       # kldivergence_loss = torch.sum(var**2+mu**2-torch.log(var)-0.5,1).mean()

       return recon_loss + variational_beta * kldivergence_loss

