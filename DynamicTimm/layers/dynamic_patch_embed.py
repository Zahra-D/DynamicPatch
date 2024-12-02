import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union
import math


class DynamicPatchEmbed(nn.Module):
    
    def __init__(
        self,
        image_size: int = 224,
        in_chans: int = 3,
        bias: bool = True,
        smallest_patch_size_to_devide: int = 14,
        embed_dim: int = 768,
        num_patches: int= 196,
        ):
        super().__init__()
        self.conv112 = nn.Conv2d(in_chans, 1, kernel_size=112, stride=112, bias=bias)
        self.conv56 = nn.Conv2d(in_chans, 1, kernel_size=56, stride=56, bias=bias)
        self.conv28 = nn.Conv2d(in_chans, 1, kernel_size=28, stride=28, bias=bias)
        self.conv14 = nn.Conv2d(in_chans, 1, kernel_size=14, stride=14, bias=bias)
        # print("hi you successfuly initiate dyanmic patch embedding in the Deit")
        
        self.selected_patches = None
    
        
        self.proj112 = nn.Conv2d(in_chans, embed_dim, kernel_size=112, stride=112, bias=bias)
        self.proj56 = nn.Conv2d(in_chans, embed_dim, kernel_size=56, stride=56, bias=bias)
        self.proj28 = nn.Conv2d(in_chans, embed_dim, kernel_size=28, stride=28, bias=bias)
        self.proj14 = nn.Conv2d(in_chans, embed_dim, kernel_size=14, stride=14, bias=bias)
        self.proj7 = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=7, bias=bias)
        # self.conv7 = nn.Conv2d(in_chans, 1, kernel_size=7, stride=7, bias=bias)
        self.activation = nn.Sigmoid()
        self.num_patches = 196  #should it be pre-defined in constructor or could be flexible and be determine in inference??? think about it later
        self.img_size = image_size

        self.sps = smallest_patch_size_to_devide
        self.mnp = (self.img_size//self.sps) #maximum number of patches in each dimension (16)
        self.num_level = int(math.log2(self.mnp))#4

        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        # self.pos_embed_generator = nn.Linear(1386, embed_dim, bias)
        self.device = None
        # self.pos_embed = nn.Parameter(torch.randn(1,1386 , embed_dim) * .02)
        
        
    def forward(self, x):
        

        
        # if self.training:            
        x= self.training_operations(x)
        # final_memory = torch.cuda.memory_allocated()
        # print(f"Final GPU Memory in forward : {final_memory / (1024**3):.2f} GB")

        # else:
        #     print('evaluation')
        #     x = self.evaluation_operations(x)
        

        return x

    def training_operations(self, x):
        # print(self.device)
        self.device = self.conv112.weight.device
        
        # print(self.device)
        
        B = x.shape[0]
        # size_scores = (self.img_size//self.smp).int()
        # num_levels = torch.log(self.mnp, self.mnp, 2)
        dividing_scores = torch.zeros((B,self.num_level+1, self.mnp, self.mnp)).to(self.device)
        
        # print('dividing score shpae', dividing_scores.shape)
        
        # y = self.conv112(x)
        dividing_scores[:,0] = self.activation(self.conv112(x)).squeeze(dim=1).repeat_interleave(2**(self.num_level-1), dim=1).repeat_interleave(2**(self.num_level-1), dim=2) +1e-8
       
        dividing_scores[:,1] = self.activation(self.conv56(x)).squeeze(dim=1).repeat_interleave(2**(self.num_level-2), dim=1).repeat_interleave(2**(self.num_level-2), dim=2) +1e-8

        dividing_scores[:,2] = self.activation(self.conv28(x)).squeeze(dim=1).repeat_interleave(2**(self.num_level-3), dim=1).repeat_interleave(2**(self.num_level-3), dim=2)+1e-8
        
        dividing_scores[:,3] = self.activation(self.conv14(x)).squeeze(dim=1) +1e-8
        
        # print('dividing score device is:', dividing_scores.device)
        
        
        
        # initial_memory = torch.cuda.memory_allocated()
        # print(f"Initial GPU Memory: {initial_memory / (1024**3):.2f} GB")

        
        selected_patches_info, S = self.selecting_patches(dividing_scores)
      
        # print('selected_patches_info score device is:', selected_patches_info.device)
        # print('S score device is:', S.device)
        flatten_index = self.flatten_index_embedings(selected_patches_info)
        all_possible_embeddings_flat = self.do_all_conv_flat(x)
        embeddings = all_possible_embeddings_flat[torch.arange(B).unsqueeze(1), flatten_index]
        
        # between_memory = torch.cuda.memory_allocated()
        # print(f"Between GPU Memory: {between_memory / (1024**3):.2f} GB")

        
        #if we decide to learn distinct pos_embed for each 1386 possible patches, then this could be moved to vit code
        # selected_patches_one_hot_S = torch.ones((B, 1386)).to(self.device)
        # selected_patches_one_hot_S.scatter_(1, flatten_index.to(self.device), S)
        # pos_structure_embed = self.pos_embed_generator(selected_patches_one_hot_S)
        # pos_embed = self.vector_projection(embeddings, pos_structure_embed)

        
        
        
        embeddings = embeddings
        
        # final_memory = torch.cuda.memory_allocated()
        # print(f"Final GPU Memory: {final_memory / (1024**3):.2f} GB")
        # print(embeddings.shape)
        return  embeddings, flatten_index, S
    
    
    def vector_projection(self, embed, pos_st):
        """
        Compute the projection of vector u onto vector v.
        """
        dot_product = torch.einsum('bpd,bd->bp', embed, pos_st)
        v_magnitude_squared = torch.einsum('bd,bd->b', pos_st, pos_st)[:, None]
        projection = torch.einsum('bp,bd->bpd' ,dot_product / (v_magnitude_squared+1e-8), pos_st)
        
        return projection
    
    
    def flatten_index_embedings(self, selected_patches_info):
        
        inner_index = (selected_patches_info[:,:-1]//selected_patches_info[:,None,-1]).int()
        w = (self.img_size//selected_patches_info[:,-1]).int()
        flatten_inner_index = inner_index[:,1] * w + inner_index[:,0]
        flatten_index = (4**(torch.log2(w))-4)//3 + flatten_inner_index
        
        return flatten_index.type(torch.int64)
            
            
        
    

    #this should be changed, not efficient by any means
    def do_all_conv_dict(self, x):
        all_possible_embeddings ={}
        all_possible_embeddings[112] = self.proj112(x)
        all_possible_embeddings[56] = self.proj56(x)
        all_possible_embeddings[28] = self.proj28(x)
        all_possible_embeddings[14] = self.proj14(x)
        all_possible_embeddings[7] = self.proj7(x) 
        
        return all_possible_embeddings
    
    
    def do_all_conv_flat(self, x):
        all_possible_embeddings = []
        all_possible_embeddings.append(self.proj112(x).view(-1,4,self.embed_dim))
        all_possible_embeddings.append(self.proj56(x).view(-1,16,self.embed_dim))
        all_possible_embeddings.append(self.proj28(x).view(-1,64,self.embed_dim))
        all_possible_embeddings.append(self.proj14(x).view(-1,256,self.embed_dim))
        all_possible_embeddings.append(self.proj7(x).view(-1,1024,self.embed_dim))
        
        return torch.concat(all_possible_embeddings, dim = 1)
        
        
        
    def calculating_flatten_index_children(self,x_p, y_p, width_p):
        
        level = torch.log2(width_p//self.sps).int()
        decision_x = x_p//self.sps
        decision_y = y_p//self.sps
        
        gap =  2**(level-1)
        if(gap<0).any():
            print(gap)

        return torch.stack([decision_y * self.mnp+decision_x, decision_y * self.mnp+decision_x+gap,
                             (decision_y+gap) * self.mnp + decision_x, (decision_y+gap) * self.mnp + decision_x + gap
                             ], dim=1) + self.mnp*self.mnp * (4 - level.unsqueeze(1) ).int()
        
        
        
    def calculating_flatten_index(self,x, y, width):
        

        
        level = torch.log2(2* width//self.sps).int()
        decision_x = x//self.sps
        decision_y = y//self.sps
    

        return (decision_y * self.mnp + decision_x + self.mnp*self.mnp * ( 4 - level )).type(torch.int64)
        
        
    
    

    def selecting_patches(self, decision_scores): 
        
        max_iter = (self.num_patches-1)//3
        B = decision_scores.shape[0]
        
        
        # print('in selecting patched function, the device of decision score is :', decision_scores.device)

        #4 layers: 0:scores value, 1:x_p  2:y_p  3:width
        scores = torch.zeros((B, 4, self.num_patches+ max_iter )).to(self.device)
        
        #shows the available options to break 
        mask_current_options = torch.zeros((B,self.num_patches + max_iter)).int().to(self.device)  
        
        x_p = torch.zeros((B)).int().to(self.device)
        y_p = torch.zeros((B)).int().to(self.device)
        width_p = self.img_size * torch.ones((B)).int().to(self.device)
        
        #setting the frist patche which is the whole image
        scores[:,0,0] = 1.0
        scores[:,1,0] = x_p
        scores[:,2,0] = y_p
        scores[:,3,0] = self.img_size
        mask_current_options[:,0] = 1

   
        i = 1
        
        
            
        for step in range(max_iter):
            
            #find the next target for breaking among all the available options
            max_indx = (scores[:,0] * mask_current_options).argmax(dim = -1)
            
            #since it is going to break, it is no more available
            mask_current_options[torch.arange(B), max_indx] = 0
            
            x_p = scores[:,1].gather(  -1, max_indx.unsqueeze(1)).squeeze().int()
            y_p = scores[:,2].gather(-1, max_indx.unsqueeze(1)).squeeze().int()
            width_p = scores[:,3].gather(-1, max_indx.unsqueeze(1)).squeeze().int()
            width_c = width_p//2
            
        
            
            sub_patches_scores_indeces = self.calculating_flatten_index_children(x_p, y_p, width_p)
            scores[:, 0,i:i+4] = decision_scores.view(B,-1)[torch.arange(B).unsqueeze(1), sub_patches_scores_indeces]
            scores[:, 1,i:i+4] = torch.stack([x_p, x_p + width_c, x_p, x_p + width_c], dim=-1)
            scores[:, 2,i:i+4] = torch.stack([y_p, y_p, y_p + width_c, y_p + width_c], dim=-1)
            scores[:, 3,i:i+4] = torch.stack([width_c]*4, dim=-1).int()
            
            #since patches with size 7 are atomic we do not consider them as available options
            mask = width_c != (self.sps //2)
            #update the options with new broken patches
            mask_current_options[:,i:i+4] = mask.unsqueeze(1).repeat_interleave(4,dim=-1)
            i += 4 # 4 new patches are added to scores
            


        mask_current_options[scores[:,3] == self.sps//2] = 1    
        
        expanded_mask = mask_current_options.unsqueeze(1)
        selected_patches = torch.masked_select(scores[:,1:], expanded_mask.bool())
        
    
        selected_patches = selected_patches.view(B, 3, self.num_patches) 
       

        
        

        #calculating scores that later will be used in loss function S_i (1-sqrt(sum(S_i)))
        cumolative_sum_decision_scores_parent = torch.cumsum(torch.concat([torch.ones(B, 1, self.mnp,self.mnp).to(self.device),
                                                                             decision_scores[:,:-1]],
                                                                            dim = 1), dim = 1)
        

        flatten_index_selected_patches= self.calculating_flatten_index(selected_patches[:,0], selected_patches[:,1],selected_patches[:,2])

        # chosen_patches_parents_S = cumolative_prod_decision_scores_parent.view(B,-1)[torch.arange(B).unsqueeze(1), flatten_index_selected_patches_parent.int()]       
        chosen_patches_parents_S = cumolative_sum_decision_scores_parent.view(B,-1).gather(dim = 1, index=flatten_index_selected_patches)     
      
          
        # flatten_index_selected_patches = self.calculating_flatten_index(selected_patches[:,0], selected_patches[:,1],selected_patches[:,2])
        # chosen_patches_S = decision_scores.view(B,-1)[torch.arange(B).unsqueeze(1), flatten_index_selected_patches.int()]        
        
        level_n = flatten_index_selected_patches//(self.mnp * self.mnp)+ 2 
        chosen_patches_S = decision_scores.view(B,-1).gather(dim = 1, index = flatten_index_selected_patches)
        selected_patches_S = ((1-chosen_patches_S) + chosen_patches_parents_S)/(level_n)
  
        
        
        self.selected_patches = [selected_patches, chosen_patches_S]

        return selected_patches, selected_patches_S


        
    # def converting_flatten_index_to_2d_index():
    #     pass
    # def geting_patches_correspond_to_flatten_index():
    #     pass


    def threshold_and_select(self, dividing_scores, K):
        dividing_decisions = {k: (A > 0.5).float() for k,A in dividing_scores.items()}
        
        selected_nodes_indeces = {}
        num_selected_nodes = 0
        B = len(dividing_decisions[112])
        parent_level_mask = torch.ones((B, 1,1))
        for (level, level_nodes), (_,level_scores) in zip(dividing_decisions.items(), dividing_scores.items()):
            
            expanded_not_selected_yet_mask_parent =  torch.repeat_interleave(torch.repeat_interleave(parent_level_mask, 2, dim=-1), 2, dim=-2)  
            # not_selected_yet_masks[level] = (torch.repeat_interleave(
            #     torch.repeat_interleave(parent_level_mask, 2, dim=-1), 2, dim=-2) * (1-level_nodes))
            
            # level_nodes_flat = level_nodes.view(-1)
            level_scores_flat = level_scores.view(-1)
            
            # when mask is one and decision is zero means it is not selected at the previous step and in this step shoul not be broken
            not_divided_indices = torch.nonzero((expanded_not_selected_yet_mask_parent * (1-level_nodes)).view(-1)).squeeze(dim=-1)

            # Sort nodes based on their original scores
            sorted_indices = torch.argsort(level_scores_flat[not_divided_indices], descending=False)

            # Select the top K nodes
            # print(zero_indices.shape)
            selected_indices = not_divided_indices[sorted_indices[:min(K-num_selected_nodes, len(not_divided_indices))]]
            num_selected_nodes += len(selected_indices)
            
            selected_nodes_indeces[level] = selected_indices
            
            if num_selected_nodes == K:
                break

            # Append selected nodes to the result
            parent_level_mask = expanded_not_selected_yet_mask_parent * level_nodes
        return selected_nodes_indeces



    def evaluation_operations(self, x):
        # Define evaluation-specific operations
        # This can be different from the training phase
        return x
        

