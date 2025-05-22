import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import argparse
from pathlib import Path
import seaborn as sns
from datetime import datetime
from sklearn.metrics import f1_score
import json
import gc # 用于垃圾回收

# 特殊Token定义
UNK_GENE_TOKEN = "<UNK>"

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_training_curves(epochs, train_metrics, val_metrics, metric_name, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
    plt.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name.capitalize()}', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(metric_name.capitalize(), fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def variable_length_collate_fn(batch):
    num_elements = len(batch[0])
    
    if num_elements == 4:
        embeddings, labels, clone_fractions, film_specific_data_batch = zip(*batch)
        has_film_specific_data = True
    elif num_elements == 3:
        embeddings, labels, clone_fractions = zip(*batch)
        film_specific_data_batch = None
        has_film_specific_data = False
    else:
        raise ValueError(f"批处理中的样本应包含3或4个元素，但得到了 {num_elements} 个。")

    max_seq_count = max(emb.shape[0] for emb in embeddings)
    batch_size = len(embeddings)
    embedding_dim = embeddings[0].shape[1]

    padded_embeddings = torch.zeros(batch_size, max_seq_count, embedding_dim)
    attention_masks = torch.zeros(batch_size, max_seq_count, dtype=torch.bool) # True for valid tokens
    padded_clone_fractions = torch.zeros(batch_size, max_seq_count)
    
    padded_film_specific_data = None
    if has_film_specific_data and film_specific_data_batch is not None:
        padded_film_specific_data = torch.zeros(batch_size, max_seq_count, dtype=torch.long)

    for i in range(batch_size):
        emb = embeddings[i]
        cf = clone_fractions[i]
        seq_count = emb.shape[0]

        padded_embeddings[i, :seq_count] = emb
        attention_masks[i, :seq_count] = True 
        if seq_count > 0:
            padded_clone_fractions[i, :seq_count] = cf
        
        if has_film_specific_data and film_specific_data_batch is not None:
            film_data_item = film_specific_data_batch[i]
            if film_data_item is not None and film_data_item.shape[0] > 0:
                 padded_film_specific_data[i, :seq_count] = film_data_item[:seq_count]

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded_embeddings, labels_tensor, padded_clone_fractions, attention_masks, padded_film_specific_data

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim_mult=4, dropout=0.1):
        super().__init__()
        hidden_dim = dim * hidden_dim_mult
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

class SingleQueryAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=2, dropout=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.zeros(1, 1, input_dim)) # Single query
        nn.init.xavier_uniform_(self.query)
        self.dropout_mod = nn.Dropout(dropout) # Renamed to avoid conflict

    def forward(self, features, key_padding_mask=None):
        batch_size = features.size(0)
        query_expanded = self.query.expand(batch_size, -1, -1)
        
        if features.size(1) == 0: # Handle empty sequences
            return torch.zeros(batch_size, features.size(2), device=features.device)

        attn_output, _ = self.self_attn(query=query_expanded, key=features, value=features, key_padding_mask=key_padding_mask, need_weights=False)
        pooled = attn_output.squeeze(1)
        pooled = self.dropout_mod(pooled)
        return pooled

class MultiQueryAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_pooling_queries=4, num_heads=2, dropout=0.2):
        super().__init__()
        self.num_pooling_queries = num_pooling_queries
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        # Multiple queries
        self.queries = nn.Parameter(torch.zeros(1, self.num_pooling_queries, input_dim))
        nn.init.xavier_uniform_(self.queries)
        self.dropout_mod = nn.Dropout(dropout) # Renamed to avoid conflict

    def forward(self, features, key_padding_mask=None):
        batch_size = features.size(0)
        queries_expanded = self.queries.expand(batch_size, -1, -1) # (B, num_pooling_queries, E)
        
        if features.size(1) == 0: # Handle empty sequences
            # Return zeros matching the expected output shape after flattening
            return torch.zeros(batch_size, self.num_pooling_queries * features.size(2), device=features.device)

        # attn_output will be (B, num_pooling_queries, E)
        attn_output, _ = self.self_attn(query=queries_expanded, key=features, value=features, key_padding_mask=key_padding_mask, need_weights=False)
        
        # Flatten the multiple query outputs: (B, num_pooling_queries * E)
        # pooled = attn_output.reshape(batch_size, -1) # Alternative way to flatten
        pooled = attn_output.contiguous().view(batch_size, -1) # Ensure contiguous before view
        pooled = self.dropout_mod(pooled)
        return pooled


class _5_6TCRRepModel_Revised(nn.Module):
    def __init__(self, embedding_dim=960, dropout=[0.2,0.2,0.4], num_classes=4,
                 num_heads=2, hidden_dim=128, num_transformer_layers=1,
                 film_condition_source=None,
                 num_gene_types=None,
                 film_mlp_internal_dim=32,
                 aggregation_type='single_query_attention', # 'single_query_attention', 'cls_token', 'multi_query_attention'
                 num_pooling_queries=4): # Relevant for multi_query_attention
        super().__init__()
        self.film_condition_source = film_condition_source
        self.hidden_dim = hidden_dim
        self.aggregation_type = aggregation_type
        self.num_pooling_queries = num_pooling_queries

        self.feat_proj_linear = nn.Linear(embedding_dim, hidden_dim)
        self.feat_proj_norm = nn.BatchNorm1d(hidden_dim)
        self.feat_proj_act = nn.GELU()
        self.feat_proj_dropout = nn.Dropout(dropout[0])
        self.cls_token = None
        if self.aggregation_type == 'cls_token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.xavier_uniform_(self.cls_token)

        self.film_gamma_beta_generator = None
        if self.film_condition_source == 'gene':
            if num_gene_types is None or num_gene_types <= 0:
                raise ValueError("num_gene_types must be provided and positive if film_condition_source is 'gene'.")
            self.film_gamma_generator_gene = nn.Embedding(num_gene_types, hidden_dim)
            self.film_beta_generator_gene = nn.Embedding(num_gene_types, hidden_dim)
        elif self.film_condition_source == 'clone_fraction':
            self.film_gamma_beta_generator_cf = nn.Sequential(
                nn.Linear(1, film_mlp_internal_dim),
                nn.GELU(),
                nn.Linear(film_mlp_internal_dim, 2 * hidden_dim)
            )

        self.transformer_layers = nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(nn.ModuleDict({
                'norm_sa': nn.LayerNorm(hidden_dim),
                'self_attn': nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout[1], batch_first=True),
                'dropout_sa': nn.Dropout(dropout[1]),
                'norm_ffn': nn.LayerNorm(hidden_dim),
                'ffn': FeedForward(hidden_dim, dropout=dropout[1]),
                'dropout_ffn': nn.Dropout(dropout[1])
            }))
        
        self.norm_pool = nn.LayerNorm(hidden_dim) # Used by all attention pooling types before pooling

        if self.aggregation_type == 'single_query_attention':
            self.pooling_module = SingleQueryAttentionPooling(hidden_dim, dropout=dropout[1], num_heads=1)
            self.class_head_input_dim = hidden_dim
        elif self.aggregation_type == 'multi_query_attention':
            self.pooling_module = MultiQueryAttentionPooling(hidden_dim, num_pooling_queries=self.num_pooling_queries, dropout=dropout[1], num_heads=1)
            self.class_head_input_dim = hidden_dim * self.num_pooling_queries
        elif self.aggregation_type == 'cls_token':
            self.pooling_module = None # CLS token is handled differently
            self.class_head_input_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported aggregation_type: {self.aggregation_type}")
        
        self.class_head = nn.Sequential(
            nn.BatchNorm1d(self.class_head_input_dim),
            nn.Linear(self.class_head_input_dim, self.class_head_input_dim // 2 if self.class_head_input_dim > 1 else 1), # Avoid dim 0
            nn.GELU(),
            nn.Dropout(dropout[2] if len(dropout) > 2 else 0.3),
            nn.Linear(self.class_head_input_dim // 2 if self.class_head_input_dim > 1 else 1, num_classes)
        )

    def forward(self, inputs, attention_mask=None, film_condition_data=None, clone_fractions_for_film=None):
        B, K_orig, _ = inputs.shape
        
        x = self.feat_proj_linear(inputs)
        if K_orig > 0:
            x_reshaped = x.reshape(B * K_orig, self.hidden_dim)
            x_bn = self.feat_proj_norm(x_reshaped)
            x = x_bn.reshape(B, K_orig, self.hidden_dim)
        x = self.feat_proj_act(x)
        x = self.feat_proj_dropout(x)

        if self.film_condition_source and K_orig > 0:
            gamma, beta = None, None
            if self.film_condition_source == 'gene':
                if film_condition_data is None: raise ValueError("film_condition_data (gene_ids) required for FiLM source 'gene'.")
                gene_ids = film_condition_data
                gamma = self.film_gamma_generator_gene(gene_ids)
                beta = self.film_beta_generator_gene(gene_ids)
            elif self.film_condition_source == 'clone_fraction':
                if clone_fractions_for_film is None: raise ValueError("clone_fractions_for_film required for FiLM source 'clone_fraction'.")
                cf_input = clone_fractions_for_film.unsqueeze(-1)
                gamma_beta = self.film_gamma_beta_generator_cf(cf_input)
                gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            if gamma is not None and beta is not None: x = gamma * x + beta
        
        final_x_for_transformer = x
        final_key_padding_mask = None # True for padded tokens

        if self.aggregation_type == 'cls_token':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            final_x_for_transformer = torch.cat((cls_tokens, x), dim=1)
            if attention_mask is not None:
                cls_mask = torch.ones(B, 1, dtype=torch.bool, device=x.device) # CLS is valid
                effective_token_mask = torch.cat((cls_mask, attention_mask), dim=1) # Valid tokens
                final_key_padding_mask = ~effective_token_mask # Padded tokens
            # Handle K_orig == 0 for CLS token
            elif K_orig == 0 and x.shape[1] == 0 : # only CLS token
                final_x_for_transformer = cls_tokens
                final_key_padding_mask = torch.zeros(B,1,dtype=torch.bool, device=x.device)


        elif attention_mask is not None: # For single_query_attention and multi_query_attention
            final_key_padding_mask = ~attention_mask # Padded tokens are True
        
        # Transformer layers
        processed_seqs = final_x_for_transformer
        for layer_module in self.transformer_layers:
            sa_input = layer_module['norm_sa'](processed_seqs)
            attn_output, _ = layer_module['self_attn'](
                query=sa_input, key=sa_input, value=sa_input,
                key_padding_mask=final_key_padding_mask,
                need_weights=False
            )
            processed_seqs = processed_seqs + layer_module['dropout_sa'](attn_output)
            ffn_input = layer_module['norm_ffn'](processed_seqs)
            ffn_output = layer_module['ffn'](ffn_input)
            processed_seqs = processed_seqs + layer_module['dropout_ffn'](ffn_output)
            
        # Aggregation
        if self.aggregation_type == 'cls_token':
            aggregated_features = processed_seqs[:, 0] # Take the CLS token output
        else: # 'single_query_attention' or 'multi_query_attention'
            # For these, processed_seqs should be the original K_orig sequences' features
            # If CLS was added, it's already part of processed_seqs, but we need non-CLS for these poolings.
            # This means the 'processed_seqs' for these poolings should come from 'x' after transformer, not 'final_x_for_transformer'
            # Correct approach: Transformer processes sequences (K_orig or 1+K_orig).
            # Then, if not CLS, take the K_orig part for pooling.
            
            if self.aggregation_type == 'cls_token': # This is already handled
                 pass
            else: # single or multi query pooling operates on sequence features only
                 # If CLS was conceptually part of transformer input, we need to extract sequence part
                 # However, our current design applies pooling AFTER the main transformer pass
                 # `processed_seqs` here is the output of transformer applied to `final_x_for_transformer`
                 # If `final_x_for_transformer` included CLS, then `processed_seqs` is (B, 1+K, D)
                 # If not, `processed_seqs` is (B, K, D)
                 
                 sequences_to_pool = processed_seqs
                 mask_for_pooling = final_key_padding_mask

                 if self.aggregation_type == 'single_query_attention' or self.aggregation_type == 'multi_query_attention':
                    # If CLS token was added to transformer input, we should pool over the actual sequence outputs
                    if self.cls_token is not None and final_x_for_transformer.shape[1] == K_orig +1 : # cls_token was used in transformer
                        sequences_to_pool = processed_seqs[:, 1:] # Exclude CLS token's output
                        if final_key_padding_mask is not None:
                             mask_for_pooling = final_key_padding_mask[:, 1:] # Get corresponding mask

                 pool_input_normed = self.norm_pool(sequences_to_pool)
                 aggregated_features = self.pooling_module(pool_input_normed, key_padding_mask=mask_for_pooling)

        return self.class_head(aggregated_features)

class TCRDataset(Dataset):
    def __init__(self, h5_path, topk=-1, topk_by='clone_fractions',
                 gene_vocab=None, unk_gene_id=None, film_condition_source=None):
        self.h5_path = h5_path
        self.sample_ids_list = []
        self.labels_list = []
        self.embeddings_list = []
        self.clone_fractions_list = [] 
        self.tcr_scores_list = []
        self.gene_ids_list = [] 
        
        self.topk = topk
        self.topk_by = topk_by
        self.film_condition_source = film_condition_source
        self.gene_vocab = gene_vocab
        self.unk_gene_id = unk_gene_id
        self._active_labels = None

        if self.film_condition_source == 'gene' and (self.gene_vocab is None or self.unk_gene_id is None):
            raise ValueError("gene_vocab and unk_gene_id must be provided if film_condition_source is 'gene'.")

        desc_suffix = ""
        if self.film_condition_source == 'gene': desc_suffix = " (FiLM: Gene)"
        elif self.film_condition_source == 'clone_fraction': desc_suffix = " (FiLM: CloneFraction)"

        print(f"从 {h5_path} 预加载数据{'(全部)' if topk == -1 else f'(每个样本前{topk}个序列，基于 {topk_by})'}{desc_suffix}...")
        
        with h5py.File(h5_path, 'r') as f:
            for sample_id in tqdm(list(f.keys()), desc=f"加载 {Path(h5_path).name}"):
                label = f[sample_id]['kmer_labels'][...][0]
                embedding = f[sample_id]['embeddings'][...]
                num_sequences_original = embedding.shape[0]

                current_clone_fraction = np.ones(num_sequences_original)
                if 'cloneFraction' in f[sample_id]: current_clone_fraction = f[sample_id]['cloneFraction'][...]
                elif self.topk_by == 'clone_fractions': print(f"警告: 样本 {sample_id} 缺少 'clone_fractions' 且被选为 topk 依据。")

                current_tcr_score = np.zeros(num_sequences_original)
                if 'tcr_scores' in f[sample_id]: current_tcr_score = f[sample_id]['tcr_scores'][...]
                elif self.topk_by == 'tcr_scores': print(f"警告: 样本 {sample_id} 缺少 'tcr_scores' 且被选为 topk 依据。")
                
                current_vhits_str_list = None
                if self.film_condition_source == 'gene':
                    if 'vhits' in f[sample_id]:
                        try:
                            current_vhits_str_list = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in f[sample_id]['vhits'][...]]
                        except Exception as e:
                            print(f"警告: 样本 {sample_id} 解码 'vhits' 失败: {e}. 将使用UNK。")
                            current_vhits_str_list = [UNK_GENE_TOKEN] * num_sequences_original
                    else:
                        print(f"警告: 样本 {sample_id} 缺少 'vhits' 且FiLM源为gene。将使用UNK。")
                        current_vhits_str_list = [UNK_GENE_TOKEN] * num_sequences_original
                
                min_len = num_sequences_original
                if len(current_clone_fraction) != num_sequences_original: min_len = min(min_len, len(current_clone_fraction))
                if len(current_tcr_score) != num_sequences_original: min_len = min(min_len, len(current_tcr_score))
                if current_vhits_str_list and len(current_vhits_str_list) != num_sequences_original: min_len = min(min_len, len(current_vhits_str_list))

                embedding = embedding[:min_len]
                current_clone_fraction = current_clone_fraction[:min_len]
                current_tcr_score = current_tcr_score[:min_len]
                if current_vhits_str_list: current_vhits_str_list = current_vhits_str_list[:min_len]

                if topk > 0 and embedding.shape[0] > topk:
                    sort_values = None
                    if self.topk_by == 'clone_fractions': sort_values = current_clone_fraction
                    elif self.topk_by == 'tcr_scores': sort_values = current_tcr_score
                    
                    if sort_values is not None and len(sort_values) == embedding.shape[0]:
                        top_indices = np.argsort(sort_values)[::-1][:topk]
                        embedding = embedding[top_indices]
                        current_clone_fraction = current_clone_fraction[top_indices]
                        current_tcr_score = current_tcr_score[top_indices]
                        if current_vhits_str_list: current_vhits_str_list = [current_vhits_str_list[i] for i in top_indices]
                    elif embedding.shape[0] > topk: 
                        embedding = embedding[:topk]
                        current_clone_fraction = current_clone_fraction[:topk]
                        current_tcr_score = current_tcr_score[:topk]
                        if current_vhits_str_list: current_vhits_str_list = current_vhits_str_list[:topk]
                
                if embedding.shape[0] == 0: 
                    print(f"警告：样本 {sample_id} 在过滤后没有序列，将被跳过。")
                    continue

                self.sample_ids_list.append(sample_id)
                self.labels_list.append(label)
                self.embeddings_list.append(embedding)
                self.clone_fractions_list.append(current_clone_fraction)
                self.tcr_scores_list.append(current_tcr_score)
                
                if self.film_condition_source == 'gene':
                    if current_vhits_str_list:
                        gene_ids_for_sample = np.array([self.gene_vocab.get(vhit, self.unk_gene_id) for vhit in current_vhits_str_list], dtype=np.int64)
                        self.gene_ids_list.append(gene_ids_for_sample)
                    else: 
                        self.gene_ids_list.append(np.full(embedding.shape[0], self.unk_gene_id, dtype=np.int64))
        
        total_seqs = sum(emb.shape[0] for emb in self.embeddings_list)
        total_samples = len(self.sample_ids_list)
        if total_samples == 0:
            print("警告：数据集中没有加载任何有效样本！请检查数据路径和topk设置。")
        else:
            print(f"已加载 {total_samples} 个样本到内存，共 {total_seqs} 条序列")
            if topk > 0: print(f"平均每个样本包含 {total_seqs/total_samples:.2f} 条序列")

    def get_active_labels(self):
        if self._active_labels is None: self._active_labels = sorted(list(np.unique(self.labels_list)))
        return self._active_labels

    def get_num_classes(self): return len(self.get_active_labels())
    def __len__(self): return len(self.sample_ids_list)

    def __getitem__(self, idx):
        item_tuple = (
            torch.from_numpy(self.embeddings_list[idx]).float(),
            torch.tensor(self.labels_list[idx], dtype=torch.long),
            torch.from_numpy(self.clone_fractions_list[idx]).float()
        )
        if self.film_condition_source == 'gene':
            item_tuple += (torch.from_numpy(self.gene_ids_list[idx]).long(),)
        return item_tuple

def plot_confusion_matrix(cm, classes, output_path, normalize=False, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)
        fmt = '.2f'
        sns.heatmap(cm_normalized, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes)
    else:
        fmt = 'd'
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=16); plt.ylabel('True Label', fontsize=14); plt.xlabel('Predict Label', fontsize=14)
    plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_roc_curve(y_true_binarized, y_score, classes, output_path, title='ROC Curve'):
    plt.figure(figsize=(10, 8)); n_classes = len(classes)
    if n_classes == 0: # No classes to plot
        print("警告: ROC曲线绘制，没有活动类别。")
        plt.close()
        return

    if n_classes == 2 and y_true_binarized.ndim == 1 and y_score.shape[1] == 2: # Binary case
         fpr, tpr, _ = roc_curve(y_true_binarized, y_score[:, 1])
         roc_auc = auc(fpr, tpr)
         plt.plot(fpr, tpr, lw=2, label=f'{classes[1]} (AUC = {roc_auc:.2f})') # Assuming classes[1] is positive
    elif y_true_binarized.ndim == 2 and y_true_binarized.shape[1] == y_score.shape[1] and y_score.shape[1] == n_classes: # Multiclass
        for i in range(n_classes):
            if np.sum(y_true_binarized[:, i]) > 0: # Check for positive samples
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
            else:
                print(f"警告: ROC曲线绘制，类别 {classes[i]} 没有正样本。")
    else:
        print(f"警告: ROC 绘制时标签维度 ({y_true_binarized.shape}), 分数维度 ({y_score.shape}), 或类别数 ({n_classes}) 不匹配。跳过ROC绘制。"); plt.close(); return

    plt.plot([0, 1], [0, 1], 'k--', lw=2); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=14); plt.ylabel('TPR', fontsize=14); plt.title(title, fontsize=16); plt.legend(loc="lower right", fontsize=12)
    
    if y_true_binarized.ndim > 1 and y_true_binarized.shape[1] > 1 and n_classes > 1: # For multi-class macro AUC
        # Check if y_true_binarized has samples for more than one class for multi_class='ovr'
        true_labels_for_auc = np.argmax(y_true_binarized, axis=1) if y_true_binarized.shape[1] > 1 else y_true_binarized
        if len(np.unique(true_labels_for_auc)) > 1:
            try:
                macro_roc_auc = roc_auc_score(y_true_binarized, y_score, average='macro', multi_class='ovr')
                plt.annotate(f'Macro Avg AUC: {macro_roc_auc:.2f}', xy=(0.5, 0.1), xycoords='axes fraction', fontsize=12)
            except ValueError as e: print(f"计算 Macro Avg AUC 时出错: {e}. 可能真实标签中只有一个类别。")
        else:
             print("警告: 计算 Macro Avg AUC 时，真实标签中只有一个类别。")
    plt.tight_layout(); plt.savefig(output_path); plt.close()


def evaluate(model, dataloader, device, label_map, criterion=None, film_condition_source=None): # Removed unused use_cls_token
    model.eval()
    all_preds_mapped_indices, all_original_labels_list, all_scores_list = [], [], []
    running_loss = 0.0
    
    if not label_map: # Handle empty label_map case
        print("警告 (evaluate): label_map 为空。")
        active_labels_sorted = []
    else:
        active_labels_sorted = sorted(label_map.keys())


    with torch.no_grad():
        for batch_data in dataloader:
            embeddings_b, original_labels_b, clone_fractions_b, attention_masks_b = batch_data[:4]
            film_specific_data_b = batch_data[4] if len(batch_data) == 5 else None

            embeddings_b, original_labels_b, clone_fractions_b, attention_masks_b = \
                embeddings_b.to(device), original_labels_b.to(device), clone_fractions_b.to(device), attention_masks_b.to(device)
            
            if film_specific_data_b is not None: 
                film_specific_data_b = film_specific_data_b.to(device)
            
            film_condition_data_for_model = None
            clone_fractions_for_film_model = None

            if film_condition_source == 'gene':
                if film_specific_data_b is None: raise ValueError("Evaluator: Gene IDs missing for FiLM source 'gene'")
                film_condition_data_for_model = film_specific_data_b
            elif film_condition_source == 'clone_fraction':
                clone_fractions_for_film_model = clone_fractions_b

            mapped_labels_b = torch.empty_like(original_labels_b, dtype=torch.long)
            if label_map: # Check if label_map is not empty
                if original_labels_b.numel() == 1: mapped_labels_b = torch.tensor([label_map[original_labels_b.item()]], device=device, dtype=torch.long)
                else: mapped_labels_b = torch.tensor([label_map[lbl.item()] for lbl in original_labels_b], device=device, dtype=torch.long)
            
            outputs = model(embeddings_b, attention_mask=attention_masks_b, 
                            film_condition_data=film_condition_data_for_model, 
                            clone_fractions_for_film=clone_fractions_for_film_model)
            
            scores = torch.softmax(outputs, dim=1)
            preds_mapped_indices_b = torch.argmax(outputs, dim=1)
            if criterion and label_map : running_loss += criterion(outputs, mapped_labels_b).item() * embeddings_b.size(0) # Check label_map
            all_preds_mapped_indices.extend(preds_mapped_indices_b.cpu().numpy())
            all_original_labels_list.extend(original_labels_b.cpu().numpy())
            all_scores_list.extend(scores.cpu().numpy())

    all_preds_mapped_indices_np, all_original_labels_np, all_scores_np = np.array(all_preds_mapped_indices), np.array(all_original_labels_list), np.array(all_scores_list)
    
    accuracy = 0.0
    macro_f1 = 0.0
    weighted_f1 = 0.0
    all_original_preds_np = np.array([])


    if len(all_original_labels_np) > 0 and label_map:
        reverse_label_map = {v: k for k, v in label_map.items()}
        # Ensure all predicted indices are in reverse_label_map before attempting conversion
        valid_preds_indices = [p_idx for p_idx in all_preds_mapped_indices_np if p_idx in reverse_label_map]
        if len(valid_preds_indices) != len(all_preds_mapped_indices_np) :
            print(f"警告: {len(all_preds_mapped_indices_np) - len(valid_preds_indices)} 个预测索引在reverse_label_map中找不到。")
        
        # Only use valid predictions for calculating accuracy and F1 scores against original labels
        # This requires careful handling if some predictions are out of map.
        # For simplicity, if any are out of map, we might not be able to calculate metrics correctly
        # Or, we could filter corresponding true labels as well.
        # Here, we assume all_preds_mapped_indices_np should correspond to keys in reverse_label_map
        try:
            all_original_preds_np = np.array([reverse_label_map[p_idx] for p_idx in all_preds_mapped_indices_np])
            accuracy = np.mean(all_original_preds_np == all_original_labels_np)
            if active_labels_sorted: # Check if list is not empty
                 macro_f1 = f1_score(all_original_labels_np, all_original_preds_np, average='macro', labels=active_labels_sorted, zero_division=0)
                 weighted_f1 = f1_score(all_original_labels_np, all_original_preds_np, average='weighted', labels=active_labels_sorted, zero_division=0)
        except KeyError as e:
            print(f"评估中发生KeyError: {e}。可能预测的索引超出了标签映射范围。")


    avg_loss_val = running_loss / len(dataloader.dataset) if criterion and dataloader.dataset and len(dataloader.dataset) > 0 else None
    return accuracy, avg_loss_val, all_original_preds_np, all_original_labels_np, all_scores_np, macro_f1, weighted_f1


def build_gene_vocab(h5_path):
    print(f"从 {h5_path} 构建基因词汇表...")
    gene_set = set()
    with h5py.File(h5_path, 'r') as f:
        for sample_id in tqdm(list(f.keys()), desc="扫描基因型"):
            if 'vhits' in f[sample_id]:
                try:
                    vhits_data = f[sample_id]['vhits'][...]
                    sample_genes = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in vhits_data]
                    gene_set.update(sample_genes)
                except Exception as e: print(f"警告: 样本 {sample_id} 读取或解码 'vhits' 失败: {e}")
    gene_vocab = {gene: i for i, gene in enumerate(sorted(list(gene_set)), 1)} 
    gene_vocab[UNK_GENE_TOKEN] = 0 
    print(f"基因词汇表构建完成，包含 {len(gene_vocab)} 个独特基因型 (包括 {UNK_GENE_TOKEN})。")
    return gene_vocab


def main(args): # args 现在是 current_args 的角色
    set_seed(args.seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # --- 这部分是原始脚本的核心输出目录逻辑，现在 'args' 代表 current_args ---
    agg_suffix = f"_{args.aggregation_type}"
    if args.aggregation_type == 'multi_query_attention':
        agg_suffix += f"_nq{args.num_pooling_queries}"

    # run_prefix 逻辑:
    # 如果是加载模型测试，则基于模型文件名；
    # 否则，如果JSON提供了run_id (通过args.run_id传入)，则使用它，
    # 最后默认为 "run"
    if args.load_model_path and Path(args.load_model_path).exists():
        run_prefix = f"test_loaded_{Path(args.load_model_path).stem}"
    elif hasattr(args, 'run_id') and args.run_id: # args.run_id 来自JSON
        run_prefix = args.run_id
    else:
        run_prefix = "run"

    film_suffix = ""
    if args.film_condition_source == 'gene':
        film_suffix = "_film_gene"
    elif args.film_condition_source == 'clone_fraction':
        film_suffix = "_film_cf"

    # 构建唯一的子目录名
    output_dir_name = f"{run_prefix}_{timestamp}{film_suffix}{agg_suffix}"

    # 最终的输出目录：在用户指定的 base_output_dir (args.output_dir from JSON) 下创建唯一子目录
    # output_dir 变量现在代表最终的、唯一的子目录路径
    output_dir = Path(args.output_dir) / output_dir_name # args.output_dir 是从JSON读取的基础目录
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- 原始输出目录逻辑结束 ---

    # 日志记录，使用 run_id (如果JSON提供) 或生成的目录名
    log_identifier = args.run_id if hasattr(args, 'run_id') and args.run_id else output_dir_name
    print(f"INFO: [{log_identifier}] 所有输出将保存到: {output_dir}")


    # 保存超参数 (hyperparameters.json)
    # 路径现在是 `output_dir / "hyperparameters.json"`，其中 output_dir 是上面生成的唯一子目录
    hyperparameters_path = output_dir / "hyperparameters.json"
    with open(hyperparameters_path, 'w') as f:
        args_dict_to_save = vars(args).copy() # 使用 vars(args) 因为 args 是 Namespace 对象
        for key, value in args_dict_to_save.items():
            if isinstance(value, Path):
                args_dict_to_save[key] = str(value)
        json.dump(args_dict_to_save, f, indent=4)
    print(f"INFO: [{log_identifier}] 超参数已保存至: {hyperparameters_path}")

    # 可视化等子目录也创建在生成的唯一 output_dir 下
    train_viz_dir = output_dir / "train_viz"
    val_viz_dir = output_dir / "val_viz"
    test_viz_dir = output_dir / "test_viz"
    test_viz_dir.mkdir(exist_ok=True); train_viz_dir.mkdir(exist_ok=True); val_viz_dir.mkdir(exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"使用设备: {device}")

    gene_vocab, num_gene_types, unk_gene_id = None, None, None
    if args.film_condition_source == 'gene':
        # Logic for loading/building gene_vocab (same as before)
        if args.load_model_path:
            try:
                checkpoint_for_vocab = torch.load(args.load_model_path, map_location='cpu')
                if 'gene_vocab' in checkpoint_for_vocab: gene_vocab = checkpoint_for_vocab['gene_vocab']
                else:
                    print(f"警告: 检查点 {args.load_model_path} 未包含 'gene_vocab'。")
                    if args.train_data and Path(args.train_data).exists():
                         print(f"尝试从 {args.train_data} 构建。")
                         gene_vocab = build_gene_vocab(args.train_data)
                    else: raise ValueError(f"FiLM gene: 检查点无词汇表且无法从训练数据构建。")
            except Exception as e:
                 print(f"从检查点加载gene_vocab失败: {e}")
                 if args.train_data and Path(args.train_data).exists():
                    print(f"尝试从 {args.train_data} 构建。")
                    gene_vocab = build_gene_vocab(args.train_data)
                 else: raise ValueError(f"FiLM gene: 检查点无词汇表且无法从训练数据构建。")
        elif args.train_data and Path(args.train_data).exists():
            gene_vocab = build_gene_vocab(args.train_data)
            with open(output_dir / f"gene_vocab{agg_suffix}.json", 'w') as fv: json.dump(gene_vocab, fv, indent=4)
        else: raise ValueError("FiLM gene: 需train_data构建词汇表或从有效load_model_path加载。")
        num_gene_types = len(gene_vocab)
        unk_gene_id = gene_vocab.get(UNK_GENE_TOKEN)
        print(f"FiLM基因词汇表大小: {num_gene_types}, UNK ID: {unk_gene_id}")

    label_map, active_labels, actual_num_classes = None, None, None
    train_dataset, val_dataset = None, None 

    if args.load_model_path is None: 
        if not args.train_data or not Path(args.train_data).exists():
            raise FileNotFoundError(f"训练模式下，训练数据文件 --train_data '{args.train_data}' 未找到或未提供。")
        train_dataset = TCRDataset(args.train_data, topk=args.topk, topk_by=args.topk_by, gene_vocab=gene_vocab, unk_gene_id=unk_gene_id, film_condition_source=args.film_condition_source)
        if len(train_dataset) == 0: raise ValueError("训练数据集为空。")
        active_labels, actual_num_classes = train_dataset.get_active_labels(), train_dataset.get_num_classes()
        if actual_num_classes == 0: raise ValueError("训练数据中没有有效的类别标签。")
        label_map = {lbl: idx for idx, lbl in enumerate(active_labels)}
        
        if args.val_data and Path(args.val_data).exists():
            val_dataset = TCRDataset(args.val_data, topk=args.topk, topk_by=args.topk_by, gene_vocab=gene_vocab, unk_gene_id=unk_gene_id, film_condition_source=args.film_condition_source)
            if len(val_dataset) == 0: print("警告：验证数据集为空。")
        else: print(f"警告：验证数据 --val_data '{args.val_data}' 未找到或未提供。")
    else: 
        print(f"加载模型模式，从 {args.load_model_path} 加载模型信息...")
        if not Path(args.load_model_path).exists(): raise FileNotFoundError(f"模型文件不存在: {args.load_model_path}")
        checkpoint = torch.load(args.load_model_path, map_location='cpu') # Ensure checkpoint is loaded once
        label_map, active_labels = checkpoint['label_map'], checkpoint['active_labels']
        actual_num_classes = len(active_labels)
    
    print(f"活动标签: {active_labels}, 类别数: {actual_num_classes}, 标签映射: {label_map if label_map else '{}'}")


    if not args.test_data or not Path(args.test_data).exists():
        raise FileNotFoundError(f"测试数据文件 --test_data '{args.test_data}' 未找到或未提供。")
    test_dataset = TCRDataset(args.test_data, topk=args.topk, topk_by=args.topk_by, gene_vocab=gene_vocab, unk_gene_id=unk_gene_id, film_condition_source=args.film_condition_source)
    if len(test_dataset) == 0: raise ValueError("测试数据集为空。")
    print(f"测试集样本数: {len(test_dataset)}")

    collate_fn_to_use = variable_length_collate_fn
    train_loader, val_loader = None, None

    if args.load_model_path is None and train_dataset:
        # Sampler logic
        sampler = None # Initialize sampler
        
        if args.use_sampler:
            print("INFO: 尝试使用加权采样器 (use_sampler=True)。")
            train_original_labels_np = np.array(train_dataset.labels_list)
            # Original sampler logic from the script, now conditional
            if active_labels and len(active_labels) > 0 : # Ensure active_labels is not empty
                # Determine bincount_size based on max label in active_labels and train_original_labels_np
                # to ensure all labels are covered.
                max_label_active = 0
                if active_labels: # active_labels contains original label values
                    max_label_active = max(active_labels)
                
                max_label_train = 0
                if len(train_original_labels_np) > 0:
                    max_label_train = np.max(train_original_labels_np)
                
                bincount_size = max(max_label_active, max_label_train) + 1
                                
                if bincount_size > 0:
                    class_sample_count = np.bincount(train_original_labels_np, minlength=bincount_size)
                    
                    # Filter active_labels to be within bincount_size to prevent IndexError,
                    # though bincount_size calculation should ideally cover this.
                    valid_active_labels_for_count = [l for l in active_labels if l < bincount_size]

                    if valid_active_labels_for_count:
                        active_class_sample_count = class_sample_count[valid_active_labels_for_count]
                        print(f"训练数据类别分布 (基于活动标签 {valid_active_labels_for_count}): {dict(zip(valid_active_labels_for_count, active_class_sample_count))}")
                        
                        if not np.all(active_class_sample_count == 0): # If any active class has samples
                            active_class_weights = 1. / (active_class_sample_count + 1e-6)
                            
                            full_class_weights = np.zeros(bincount_size)
                            # Assign weights to the positions corresponding to valid_active_labels
                            for label, weight in zip(valid_active_labels_for_count, active_class_weights):
                                full_class_weights[label] = weight
                            
                            # Ensure all labels in train_original_labels_np are within bounds of full_class_weights
                            # This should be true if bincount_size is correct.
                            sample_weights = full_class_weights[train_original_labels_np]
                            
                            if np.sum(sample_weights) > 0: # Check if any sample got a non-zero weight
                                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
                                print("INFO: 加权采样器已启用。")
                            else:
                                print("警告: 计算出的样本权重总和为0。将使用标准随机打乱。")
                        else: 
                            print("警告: 所有活动类别的样本数均为0，无法计算权重。将使用标准随机打乱。")
                    else:
                        print("警告: 没有有效的活动标签用于计算权重。将使用标准随机打乱。")
                else: 
                    print("警告: 无法确定bincount_size进行加权采样 (bincount_size=0)。将使用标准随机打乱。")
            else: 
                print("警告: 无活动标签或训练标签列表为空，无法进行加权采样。将使用标准随机打乱。")
        else: # args.use_sampler is False
            print("INFO: 未使用加权采样器 (use_sampler=False)。将使用标准随机打乱。")
            # sampler remains None, and shuffle will be True for the DataLoader
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  sampler=sampler, shuffle=(sampler is None), # shuffle is True if sampler is None, False otherwise
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_to_use)
        if val_dataset and len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_to_use)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn_to_use)
    
    embedding_dim_source_h5 = next((p for p in [args.train_data, args.val_data, args.test_data] if p and Path(p).exists()), None)
    if not embedding_dim_source_h5: raise FileNotFoundError("无法找到任何有效的H5文件来确定嵌入维度。")
    with h5py.File(embedding_dim_source_h5, 'r') as f:
        first_sample_key = next(iter(f.keys()))
        try:
            embedding_dim = f[first_sample_key]['embeddings'].shape[1]
            if f[first_sample_key]['embeddings'].shape[0] == 0: # if first sample is empty
                for key in f.keys():
                    if 'embeddings' in f[key] and f[key]['embeddings'].shape[0] > 0:
                        embedding_dim = f[key]['embeddings'].shape[1]; break
                else: raise ValueError("所有样本的嵌入都为空。")
        except Exception as e: raise ValueError(f"从 {embedding_dim_source_h5} 确定嵌入维度失败: {e}")
    print(f"从 {embedding_dim_source_h5} 确定的嵌入维度: {embedding_dim}")

    model = _5_6TCRRepModel_Revised(
        embedding_dim=embedding_dim, num_classes=actual_num_classes, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, num_transformer_layers=args.num_transformer_layers,
        film_condition_source=args.film_condition_source,
        num_gene_types=num_gene_types,
        film_mlp_internal_dim=args.film_mlp_internal_dim,
        aggregation_type=args.aggregation_type, # Pass aggregation type
        num_pooling_queries=args.num_pooling_queries, # Pass num queries
        dropout=args.dropout
    ).to(device)

    if args.load_model_path:
        print(f"从 {args.load_model_path} 加载模型状态...")
        # checkpoint was loaded earlier if in load_model_path mode
        agg_type_in_checkpoint = checkpoint.get('aggregation_type', 'single_query_attention') # Default if not found
        num_queries_in_checkpoint = checkpoint.get('num_pooling_queries', 4)

        if agg_type_in_checkpoint != args.aggregation_type or \
           (args.aggregation_type == 'multi_query_attention' and num_queries_in_checkpoint != args.num_pooling_queries):
            print(f"警告: 命令行聚合参数与检查点不符。命令行: {args.aggregation_type}, NQ:{args.num_pooling_queries}。检查点: {agg_type_in_checkpoint}, NQ:{num_queries_in_checkpoint}。将按命令行参数加载结构。")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print("模型加载完成，跳过训练。")
        test_acc, _, test_preds, test_labels, test_scores, test_macro_f1, test_weighted_f1 = evaluate(
            model, test_loader, device, label_map, film_condition_source=args.film_condition_source
        )
        print(f"加载的模型 - 测试集准确率: {test_acc:.4f}, 宏F1: {test_macro_f1:.4f}, 加权F1: {test_weighted_f1:.4f}")
        
        original_class_names_map = checkpoint.get('original_class_names_map', {0:'0',1:'1',2:'2',3:'3',4:'4'}) # default map
        active_class_names = [original_class_names_map.get(i, f"未知{i}") for i in active_labels] if active_labels else ["N/A"]
        
        if len(test_labels) > 0 and active_labels: # Ensure data for CM
            test_cm = confusion_matrix(test_labels, test_preds, labels=active_labels)
            plot_confusion_matrix(test_cm, active_class_names, test_viz_dir / f"test_cm_loaded_{Path(args.load_model_path).stem}{film_suffix}{agg_suffix}.png", title=f'Test CM (Loaded) ACC:{test_acc:.4f}')
        
            if actual_num_classes > 0 and len(test_scores) > 0: # Ensure data for ROC
                test_labels_bin = label_binarize(test_labels, classes=active_labels)
                if test_labels_bin.ndim == 1 and actual_num_classes == 2: pass # Binary
                elif test_labels_bin.shape[1] != actual_num_classes and not (actual_num_classes == 2 and test_labels_bin.shape[1] == 1 and test_scores.shape[1] == 2):
                     print(f"警告: 加载模型测试ROC, 标签二值化列数 {test_labels_bin.shape[1]} 与 分数/类别数 {test_scores.shape[1]}/{actual_num_classes} 不匹配。")
                else:
                     plot_roc_curve(test_labels_bin, test_scores, active_class_names, test_viz_dir / f"test_roc_loaded_{Path(args.load_model_path).stem}{film_suffix}{agg_suffix}.png", title=f'Test ROC (Loaded)')
        else:
            print("加载的模型：测试集标签为空或无活动标签，跳过混淆矩阵和ROC曲线绘制。")


        test_results_loaded = {
            'accuracy': test_acc, 'macro_f1': test_macro_f1, 'weighted_f1': test_weighted_f1,
            'confusion_matrix': test_cm.tolist() if 'test_cm' in locals() else [], 
            'predictions': test_preds.tolist(), 'labels': test_labels.tolist(), 'scores': test_scores.tolist(),
            'active_labels': active_labels, 'active_class_names': active_class_names,
            'loaded_model_path': str(args.load_model_path), # Path to string
            'hyperparameters_args': vars(args)
        }
        # Convert Path objects in args to strings for saving
        for key, value in test_results_loaded['hyperparameters_args'].items():
            if isinstance(value, Path):
                test_results_loaded['hyperparameters_args'][key] = str(value)
        np.save(output_dir / f"test_results_loaded_{Path(args.load_model_path).stem}{film_suffix}{agg_suffix}.npy", test_results_loaded)
        print(f"加载模型的测试结果已保存至: {output_dir / f'test_results_loaded_{Path(args.load_model_path).stem}{film_suffix}{agg_suffix}.npy'}")

    else: # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        history = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'train_macro_f1': [], 'val_macro_f1': []}
        
        original_class_names_map = {0:'0',1:'1',2:'2',3:'3',4:'4'} # Example, make configurable
        active_class_names = [original_class_names_map.get(i, f"未知{i}") for i in active_labels] if active_labels else ["N/A"]

        best_val_f1 = -1.0 
        best_val_acc = -1.0 
        best_model_f1_path = output_dir / f"best_model_f1{film_suffix}{agg_suffix}.pth" # 作为默认或占位符
        best_model_acc_path = output_dir / f"best_model_acc{film_suffix}{agg_suffix}.pth" # 作为默认或占位符

        print("开始训练...")
        if train_loader is None: raise ValueError("训练加载器未初始化。")

        for epoch in range(args.epochs):
            model.train()
            train_preds_orig, train_labels_orig, train_scores_list = [], [], []
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for batch_data_train in pbar:
                # (Same batch processing as before)
                embeddings_b, original_labels_b, clone_fractions_b, attention_masks_b = batch_data_train[:4]
                film_specific_data_b = batch_data_train[4] if len(batch_data_train) == 5 else None

                embeddings_b,original_labels_b,clone_fractions_b,attention_masks_b = embeddings_b.to(device),original_labels_b.to(device),clone_fractions_b.to(device),attention_masks_b.to(device)
                
                if film_specific_data_b is not None: film_specific_data_b = film_specific_data_b.to(device)

                film_condition_data_for_model_train = None
                clone_fractions_for_film_model_train = None
                if args.film_condition_source == 'gene':
                    if film_specific_data_b is None: raise ValueError("Train: Gene IDs missing for FiLM gene")
                    film_condition_data_for_model_train = film_specific_data_b
                elif args.film_condition_source == 'clone_fraction':
                    clone_fractions_for_film_model_train = clone_fractions_b 

                mapped_labels_b = torch.empty_like(original_labels_b, dtype=torch.long)
                if label_map: # Check if label_map is not empty
                    if original_labels_b.numel()==1: mapped_labels_b = torch.tensor([label_map[original_labels_b.item()]], device=device,dtype=torch.long)
                    else: mapped_labels_b = torch.tensor([label_map[lbl.item()] for lbl in original_labels_b], device=device,dtype=torch.long)
                
                optimizer.zero_grad()
                outputs = model(embeddings_b, attention_mask=attention_masks_b, film_condition_data=film_condition_data_for_model_train, clone_fractions_for_film=clone_fractions_for_film_model_train)
                
                loss = torch.tensor(0.0, device=device) # Default loss
                if label_map and mapped_labels_b.numel() > 0 : # Check if there are labels to compute loss
                     loss = criterion(outputs, mapped_labels_b)
                     loss.backward()
                     optimizer.step()
                running_loss += loss.item() * embeddings_b.size(0)
                
                scores_b = torch.softmax(outputs, dim=1)
                preds_mapped_indices_b = torch.argmax(outputs, dim=1)
                
                if label_map: # Check if label_map is not empty
                    reverse_label_map = {v: k for k, v in label_map.items()}
                    try:
                        original_preds_b = torch.tensor([reverse_label_map[p.item()] for p in preds_mapped_indices_b], device=device, dtype=torch.long) # Should be .cpu() before list
                        train_preds_orig.extend(original_preds_b.cpu().numpy())
                    except KeyError: # Handle cases where pred_mapped_index is not in reverse_label_map (e.g. if model predicts outside defined classes)
                         # print(f"Warning: Predicted index not in reverse_label_map during training batch.")
                         # Add placeholder or skip, here we add what was predicted directly from argmax
                         train_preds_orig.extend(preds_mapped_indices_b.cpu().numpy())


                train_labels_orig.extend(original_labels_b.cpu().numpy())
                train_scores_list.extend(scores_b.detach().cpu().numpy())
                pbar.set_postfix({'loss': loss.item()})

            epoch_loss = running_loss / len(train_loader.dataset) if train_loader.dataset and len(train_loader.dataset) > 0 else 0
            train_preds_np, train_labels_np = np.array(train_preds_orig), np.array(train_labels_orig)
            
            train_acc = 0.0; train_macro_f1 = 0.0
            if len(train_labels_np) > 0 and len(train_preds_np) == len(train_labels_np): # Ensure alignment
                train_acc = np.mean(train_preds_np == train_labels_np)
                if active_labels: # Check if not empty
                    train_macro_f1 = f1_score(train_labels_np, train_preds_np, average='macro', labels=active_labels, zero_division=0)
            
            val_acc, val_loss, val_preds, val_labels, val_scores, val_macro_f1, _ = (0,0,[],[],[],0,0) 
            if val_loader and val_loader.dataset and len(val_loader.dataset) > 0 :
                 val_acc, val_loss, val_preds, val_labels, val_scores, val_macro_f1, _ = evaluate(
                     model, val_loader, device, label_map, criterion, 
                     film_condition_source=args.film_condition_source
                )
            
            history['epochs'].append(epoch+1); history['train_loss'].append(epoch_loss); history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss if val_loss is not None else 0); history['val_acc'].append(val_acc)
            history['train_macro_f1'].append(train_macro_f1); history['val_macro_f1'].append(val_macro_f1)
            print(f"E{epoch+1} - TLs:{epoch_loss:.4f} TAc:{train_acc:.4f} TMF1:{train_macro_f1:.4f} | VLs:{val_loss if val_loss is not None else -1:.4f} VAc:{val_acc:.4f} VMF1:{val_macro_f1:.4f}")

            # Plotting (same logic, ensure active_class_names is valid)
            if len(train_labels_np) > 0 and active_labels and len(train_preds_np) == len(train_labels_np):
                cm_train = confusion_matrix(train_labels_np, train_preds_np, labels=active_labels)
                plot_confusion_matrix(cm_train, active_class_names, train_viz_dir / f"train_cm_e{epoch+1}{film_suffix}{agg_suffix}.png", title=f'Train CM E{epoch+1} ACC:{train_acc:.4f}')
            if val_loader and len(val_labels) > 0 and active_labels:
                cm_val = confusion_matrix(val_labels, val_preds, labels=active_labels)
                plot_confusion_matrix(cm_val, active_class_names, val_viz_dir / f"val_cm_e{epoch+1}{film_suffix}{agg_suffix}.png", title=f'Val CM E{epoch+1} ACC:{val_acc:.4f}')
            
            if actual_num_classes > 0 and active_labels and len(active_labels) > 0: # Check active_labels
                if len(train_labels_np) > 0 and len(train_scores_list) == len(train_labels_np):
                    train_labels_bin = label_binarize(train_labels_np, classes=active_labels)
                    train_scores_np = np.array(train_scores_list)
                    if train_labels_bin.ndim == 1 and actual_num_classes == 2 :  pass 
                    elif train_labels_bin.shape[1] != train_scores_np.shape[1] and not (actual_num_classes == 2 and train_scores_np.shape[1]==2 and train_labels_bin.shape[1]==1):
                        print(f"警告: 训练集 ROC, 标签二值化列数 {train_labels_bin.shape[1]} 与 分数/类别数 {train_scores_np.shape[1]}/{actual_num_classes} 不匹配。")
                    else:
                        plot_roc_curve(train_labels_bin, train_scores_np, active_class_names, train_viz_dir / f"train_roc_e{epoch+1}{film_suffix}{agg_suffix}.png", title=f'Train ROC E{epoch+1}')
                
                if val_loader and len(val_labels) > 0 and len(val_scores) == len(val_labels): 
                    val_labels_bin = label_binarize(val_labels, classes=active_labels)
                    if val_labels_bin.ndim == 1 and actual_num_classes == 2: pass
                    elif val_labels_bin.shape[1] != val_scores.shape[1] and not (actual_num_classes == 2 and val_scores.shape[1]==2 and val_labels_bin.shape[1]==1):
                         print(f"警告: 验证集 ROC, 标签二值化列数 {val_labels_bin.shape[1]} 与 分数/类别数 {val_scores.shape[1]}/{actual_num_classes} 不匹配。")
                    else:
                        plot_roc_curve(val_labels_bin, val_scores, active_class_names, val_viz_dir / f"val_roc_e{epoch+1}{film_suffix}{agg_suffix}.png", title=f'Val ROC E{epoch+1}')

            if len(history['epochs']) > 1:
                plot_training_curves(history['epochs'], history['train_acc'], history['val_acc'], 'accuracy', output_dir / f'accuracy_curve{film_suffix}{agg_suffix}.png')
                plot_training_curves(history['epochs'], history['train_loss'], history['val_loss'], 'loss', output_dir / f'loss_curve{film_suffix}{agg_suffix}.png')
                plot_training_curves(history['epochs'], history['train_macro_f1'], history['val_macro_f1'], 'macro_f1', output_dir / f'macro_f1_curve{film_suffix}{agg_suffix}.png')

            save_payload = {
                'epoch': epoch, 'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                'val_acc': val_acc, 'val_macro_f1': val_macro_f1, 
                'label_map': label_map, 'active_labels': active_labels,
                'film_condition_source': args.film_condition_source, 
                'aggregation_type': args.aggregation_type, # Save aggregation type
                'num_pooling_queries': args.num_pooling_queries, # Save num queries
                'dropout_args': args.dropout, 
                'hidden_dim_args': args.hidden_dim, 
                'num_heads_args': args.num_heads, 
                'num_transformer_layers_args': args.num_transformer_layers,
                'original_class_names_map': original_class_names_map
            }
            if args.film_condition_source == 'gene': save_payload.update({'gene_vocab': gene_vocab, 'num_gene_types': num_gene_types})
            elif args.film_condition_source == 'clone_fraction': save_payload['film_mlp_internal_dim'] = args.film_mlp_internal_dim
            
            current_best_f1_metric = val_macro_f1 if val_loader and val_loader.dataset and len(val_loader.dataset) > 0 else train_macro_f1
            current_best_acc_metric = val_acc if val_loader and val_loader.dataset and len(val_loader.dataset) > 0 else train_acc
            
            if current_best_f1_metric > best_val_f1:
                best_val_f1 = current_best_f1_metric
                # 更新文件名以包含 epoch 信息
                epoch_specific_f1_path = output_dir / f"best_model_f1_e{epoch+1}{film_suffix}{agg_suffix}.pth"
                torch.save(save_payload, epoch_specific_f1_path)
                best_model_f1_path = epoch_specific_f1_path # 更新 best_model_f1_path 指向新的带epoch的文件
                print(f"保存最佳F1模型至 {best_model_f1_path} (F1: {current_best_f1_metric:.4f}, Acc: {current_best_acc_metric:.4f})")
            
            if current_best_acc_metric > best_val_acc:
                best_val_acc = current_best_acc_metric
                # 更新文件名以包含 epoch 信息
                epoch_specific_acc_path = output_dir / f"best_model_acc_e{epoch+1}{film_suffix}{agg_suffix}.pth"
                torch.save(save_payload, epoch_specific_acc_path)
                best_model_acc_path = epoch_specific_acc_path # 更新 best_model_acc_path 指向新的带epoch的文件
                print(f"保存最佳准确率模型至 {best_model_acc_path} (Acc: {current_best_acc_metric:.4f}, F1: {current_best_f1_metric:.4f})")

        print(f"训练完成！最佳{'验证' if val_loader and val_loader.dataset and len(val_loader.dataset)>0 else '训练'}宏平均F1: {best_val_f1:.4f}, 最佳准确率: {best_val_acc:.4f}")
        
        # Test with best F1 model
        if best_model_f1_path.exists():
            print(f"\n--- 使用训练后的最佳F1模型 ({best_model_f1_path.name}) 进行测试 ---")
            checkpoint_best_f1 = torch.load(best_model_f1_path, map_location=device)
            # Reload model with parameters from checkpoint
            model_best_f1 = _5_6TCRRepModel_Revised(
                embedding_dim=embedding_dim, num_classes=len(checkpoint_best_f1['active_labels']), 
                hidden_dim=checkpoint_best_f1.get('hidden_dim_args', args.hidden_dim), 
                num_heads=checkpoint_best_f1.get('num_heads_args', args.num_heads), 
                num_transformer_layers=checkpoint_best_f1.get('num_transformer_layers_args', args.num_transformer_layers), 
                film_condition_source=checkpoint_best_f1.get('film_condition_source'), 
                num_gene_types=checkpoint_best_f1.get('num_gene_types'), 
                film_mlp_internal_dim=checkpoint_best_f1.get('film_mlp_internal_dim', args.film_mlp_internal_dim),
                dropout=checkpoint_best_f1.get('dropout_args', args.dropout),
                aggregation_type=checkpoint_best_f1.get('aggregation_type', args.aggregation_type),
                num_pooling_queries=checkpoint_best_f1.get('num_pooling_queries', args.num_pooling_queries)
            ).to(device)
            model_best_f1.load_state_dict(checkpoint_best_f1['model_state_dict'])
            
            test_acc_bf1, _, test_preds_bf1, test_labels_bf1, test_scores_bf1, test_mf1_bf1, test_wf1_bf1 = evaluate(
                model_best_f1, test_loader, device, checkpoint_best_f1['label_map'], 
                film_condition_source=checkpoint_best_f1.get('film_condition_source')
            )
            print(f"最佳F1模型 - 测试集准确率: {test_acc_bf1:.4f}, 宏F1: {test_mf1_bf1:.4f}, 加权F1: {test_wf1_bf1:.4f}")
            
            loaded_original_class_names_map_f1 = checkpoint_best_f1.get('original_class_names_map', original_class_names_map)
            active_class_names_bf1 = [loaded_original_class_names_map_f1.get(i, f"未知{i}") for i in checkpoint_best_f1['active_labels']] if checkpoint_best_f1.get('active_labels') else ["N/A"]

            if len(test_labels_bf1) > 0 and checkpoint_best_f1.get('active_labels'):
                test_cm_bf1 = confusion_matrix(test_labels_bf1, test_preds_bf1, labels=checkpoint_best_f1['active_labels'])
                plot_confusion_matrix(test_cm_bf1, active_class_names_bf1, test_viz_dir / f"test_cm_best_f1{film_suffix}{agg_suffix}.png", title=f'Test CM (Best F1 Model) ACC:{test_acc_bf1:.4f}')
            
                num_classes_bf1 = len(checkpoint_best_f1['active_labels'])
                if num_classes_bf1 > 0 and len(test_scores_bf1) == len(test_labels_bf1):
                    test_labels_bin_bf1 = label_binarize(test_labels_bf1, classes=checkpoint_best_f1['active_labels'])
                    if test_labels_bin_bf1.ndim == 1 and num_classes_bf1 == 2: pass
                    elif test_labels_bin_bf1.shape[1] != test_scores_bf1.shape[1] and not (num_classes_bf1 == 2 and test_scores_bf1.shape[1]==2 and test_labels_bin_bf1.shape[1]==1):
                        print(f"警告: 最佳F1模型测试ROC, 标签二值化列数 {test_labels_bin_bf1.shape[1]} 与 分数/类别数 {test_scores_bf1.shape[1]}/{num_classes_bf1} 不匹配。")
                    else:
                        plot_roc_curve(test_labels_bin_bf1, test_scores_bf1, active_class_names_bf1, test_viz_dir / f"test_roc_best_f1{film_suffix}{agg_suffix}.png", title=f'Test ROC (Best F1 Model)')
            
            test_results_best_f1 = {
                'accuracy': test_acc_bf1, 'macro_f1': test_mf1_bf1, 'weighted_f1': test_wf1_bf1,
                'confusion_matrix': test_cm_bf1.tolist() if 'test_cm_bf1' in locals() else [], 
                'predictions': test_preds_bf1.tolist(), 'labels': test_labels_bf1.tolist(), 
                'scores': test_scores_bf1.tolist(),
                'active_labels': checkpoint_best_f1['active_labels'], 
                'active_class_names': active_class_names_bf1,
                'model_path': str(best_model_f1_path),
                'hyperparameters_args': vars(args)
            }
            for key, value in test_results_best_f1['hyperparameters_args'].items(): # Path to str
                if isinstance(value, Path): test_results_best_f1['hyperparameters_args'][key] = str(value)
            np.save(output_dir / f"test_results_best_f1{film_suffix}{agg_suffix}.npy", test_results_best_f1)
        else: print(f"未找到最佳F1模型文件: {best_model_f1_path}")

        # Test with best Accuracy model (similar logic as best F1)
        if best_model_acc_path.exists():
            print(f"\n--- 使用训练后的最佳准确率模型 ({best_model_acc_path.name}) 进行测试 ---")
            checkpoint_best_acc = torch.load(best_model_acc_path, map_location=device)
            model_best_acc = _5_6TCRRepModel_Revised( # Instantiate with checkpoint params
                embedding_dim=embedding_dim, num_classes=len(checkpoint_best_acc['active_labels']), 
                hidden_dim=checkpoint_best_acc.get('hidden_dim_args', args.hidden_dim), 
                num_heads=checkpoint_best_acc.get('num_heads_args', args.num_heads), 
                num_transformer_layers=checkpoint_best_acc.get('num_transformer_layers_args', args.num_transformer_layers), 
                film_condition_source=checkpoint_best_acc.get('film_condition_source'), 
                num_gene_types=checkpoint_best_acc.get('num_gene_types'), 
                film_mlp_internal_dim=checkpoint_best_acc.get('film_mlp_internal_dim', args.film_mlp_internal_dim),
                dropout=checkpoint_best_acc.get('dropout_args', args.dropout),
                aggregation_type=checkpoint_best_acc.get('aggregation_type', args.aggregation_type),
                num_pooling_queries=checkpoint_best_acc.get('num_pooling_queries', args.num_pooling_queries)
            ).to(device)
            model_best_acc.load_state_dict(checkpoint_best_acc['model_state_dict'])

            test_acc_bacc, _, test_preds_bacc, test_labels_bacc, test_scores_bacc, test_mf1_bacc, test_wf1_bacc = evaluate(
                model_best_acc, test_loader, device, checkpoint_best_acc['label_map'], 
                film_condition_source=checkpoint_best_acc.get('film_condition_source')
            )
            print(f"最佳准确率模型 - 测试集准确率: {test_acc_bacc:.4f}, 宏F1: {test_mf1_bacc:.4f}, 加权F1: {test_wf1_bacc:.4f}")
            
            loaded_original_class_names_map_acc = checkpoint_best_acc.get('original_class_names_map', original_class_names_map)
            active_class_names_bacc = [loaded_original_class_names_map_acc.get(i, f"未知{i}") for i in checkpoint_best_acc['active_labels']] if checkpoint_best_acc.get('active_labels') else ["N/A"]

            if len(test_labels_bacc) > 0 and checkpoint_best_acc.get('active_labels'):
                test_cm_bacc = confusion_matrix(test_labels_bacc, test_preds_bacc, labels=checkpoint_best_acc['active_labels'])
                plot_confusion_matrix(test_cm_bacc, active_class_names_bacc, test_viz_dir / f"test_cm_best_acc{film_suffix}{agg_suffix}.png", title=f'Test CM (Best Acc Model) ACC:{test_acc_bacc:.4f}')
            
                num_classes_bacc = len(checkpoint_best_acc['active_labels'])
                if num_classes_bacc > 0 and len(test_scores_bacc) == len(test_labels_bacc):
                    test_labels_bin_bacc = label_binarize(test_labels_bacc, classes=checkpoint_best_acc['active_labels'])
                    if test_labels_bin_bacc.ndim == 1 and num_classes_bacc == 2: pass
                    elif test_labels_bin_bacc.shape[1] != test_scores_bacc.shape[1] and not (num_classes_bacc == 2 and test_scores_bacc.shape[1]==2 and test_labels_bin_bacc.shape[1]==1):
                        print(f"警告: 最佳Acc模型测试ROC, 标签二值化列数 {test_labels_bin_bacc.shape[1]} 与 分数/类别数 {test_scores_bacc.shape[1]}/{num_classes_bacc} 不匹配。")
                    else:
                        plot_roc_curve(test_labels_bin_bacc, test_scores_bacc, active_class_names_bacc, test_viz_dir / f"test_roc_best_acc{film_suffix}{agg_suffix}.png", title=f'Test ROC (Best Acc Model)')
            
            test_results_best_acc = {
                'accuracy': test_acc_bacc, 'macro_f1': test_mf1_bacc, 'weighted_f1': test_wf1_bacc,
                'confusion_matrix': test_cm_bacc.tolist() if 'test_cm_bacc' in locals() else [], 
                'predictions': test_preds_bacc.tolist(), 'labels': test_labels_bacc.tolist(), 
                'scores': test_scores_bacc.tolist(),
                'active_labels': checkpoint_best_acc['active_labels'], 
                'active_class_names': active_class_names_bacc,
                'model_path': str(best_model_acc_path),
                'hyperparameters_args': vars(args)
            }
            for key, value in test_results_best_acc['hyperparameters_args'].items(): # Path to str
                if isinstance(value, Path): test_results_best_acc['hyperparameters_args'][key] = str(value)
            np.save(output_dir / f"test_results_best_acc{film_suffix}{agg_suffix}.npy", test_results_best_acc)
        else: print(f"未找到最佳准确率模型文件: {best_model_acc_path}")

        print(f"\n训练和测试结果已保存至: {output_dir}")


if __name__ == "__main__":
    # 1. 新的 ArgumentParser，只接收 JSON 配置文件路径
    cli_parser = argparse.ArgumentParser(description='通过JSON配置训练TCR序列表示模型')
    cli_parser.add_argument('--config_path', type=str, default="/xiongjun/test/NEW-MIL-dir-backup/backup/NEW-MIL-dir-backup/exp_json/521-exp-setup/521setup-p3.json",
                            help='包含一个或多个参数配置的JSON文件路径')
    
    cli_args = cli_parser.parse_args()

    # 2. 加载 JSON 配置文件
    try:
        with open(cli_args.config_path, 'r') as f:
            all_configurations = json.load(f)
        if not isinstance(all_configurations, list):
            print(f"错误: JSON配置文件 {cli_args.config_path} 的顶层应为一个列表 ([...])。")
            exit(1)
    except FileNotFoundError:
        print(f"错误: JSON配置文件未找到: {cli_args.config_path}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: 解析JSON配置文件 {cli_args.config_path} 失败: {e}")
        exit(1)


    # 3. 循环遍历所有配置并执行
    for i, config_params_from_json in enumerate(all_configurations):
        if not isinstance(config_params_from_json, dict):
            print(f"警告: JSON配置文件中的第 {i+1} 项不是一个有效的配置字典。已跳过。")
            continue

        run_id_for_logging = config_params_from_json.get('run_id', f'json_config_{i+1}')
        print(f"\n{'='*30} 开始运行配置 {i+1}/{len(all_configurations)}: {run_id_for_logging} {'='*30}")

        # 4. 创建一个临时的 ArgumentParser 来获取原始脚本的默认参数
        #    这与您原始脚本末尾的 parser 定义几乎一致
        original_defaults_parser = argparse.ArgumentParser(add_help=False) # add_help=False 避免冲突

        # --- 从您原始脚本复制所有 add_argument 调用到这里，以获取默认值 ---
        original_defaults_parser.add_argument('--train_data', type=str, default=None) # 示例，添加所有原始参数
        original_defaults_parser.add_argument('--val_data', type=str, default=None)
        original_defaults_parser.add_argument('--test_data', type=str, default=None)
        # !! 注意: output_dir 的默认值应该是基础目录的默认值
        original_defaults_parser.add_argument('--output_dir', type=str, default=f"./experiment_outputs_base_{datetime.now().strftime('%Y%m%d')}")
        original_defaults_parser.add_argument('--film_condition_source', type=str, default=None, choices=[None, 'gene', 'clone_fraction'])
        original_defaults_parser.add_argument('--film_mlp_internal_dim', type=int, default=32)
        original_defaults_parser.add_argument('--aggregation_type', type=str, default='multi_query_attention', 
                                    choices=['single_query_attention', 'cls_token', 'multi_query_attention'])
        original_defaults_parser.add_argument('--num_pooling_queries', type=int, default=4)
        original_defaults_parser.add_argument('--batch_size', type=int, default=8)
        original_defaults_parser.add_argument('--topk', type=int, default=5000)
        original_defaults_parser.add_argument('--topk_by', type=str, default='tcr_scores', choices=['clone_fractions', 'tcr_scores'])
        original_defaults_parser.add_argument('--learning_rate', type=float, default=1e-4)
        original_defaults_parser.add_argument('--weight_decay', type=float, default=1e-4)
        original_defaults_parser.add_argument('--use_sampler', default=True)
        original_defaults_parser.add_argument('--epochs', type=int, default=100) 
        original_defaults_parser.add_argument('--dropout', type=float, nargs='+', default=[0.3, 0.2, 0.5])
        original_defaults_parser.add_argument('--hidden_dim', type=int, default=256)
        original_defaults_parser.add_argument('--num_heads', type=int, default=2)
        original_defaults_parser.add_argument('--num_transformer_layers', type=int, default=1)
        original_defaults_parser.add_argument('--num_workers', type=int, default=0)
        original_defaults_parser.add_argument('--gpu', type=int, default=0)
        original_defaults_parser.add_argument('--seed', type=int, default=42)
        original_defaults_parser.add_argument('--load_model_path', type=str, default=None)
        # 新增一个 run_id 参数的默认定义，以便在 main 函数中可以安全访问
        # 即使 JSON 中未提供 run_id，也会有一个默认值
        original_defaults_parser.add_argument('--run_id', type=str, default=f"cfg_run_{i+1}")
        # --- 原始参数定义结束 ---

        # 解析空列表以获取包含所有默认值的 Namespace 对象
        default_args_namespace, _ = original_defaults_parser.parse_known_args([])
        
        # 将默认参数转换为字典
        current_run_params_dict = vars(default_args_namespace).copy()
        
        # 使用 JSON 中的参数覆盖默认值
        current_run_params_dict.update(config_params_from_json)
        
        # 创建最终的 Namespace 对象传递给 main 函数
        args_for_run = argparse.Namespace(**current_run_params_dict)

        # 参数校验 (与原始脚本类似，但针对 args_for_run)
        # 例如：dropout 长度校验
        if not isinstance(args_for_run.dropout, list) or len(args_for_run.dropout) != 3:
            if isinstance(args_for_run.dropout, str): # 尝试从字符串 "0.3 0.2 0.5" 转换
                try: 
                    args_for_run.dropout = [float(x.strip()) for x in args_for_run.dropout.split()]
                except ValueError: pass # 如果转换失败，则下面会用默认值
            
            if not isinstance(args_for_run.dropout, list) or len(args_for_run.dropout) != 3:
                print(f"警告: 配置 {run_id_for_logging} 中的 'dropout' 参数格式不正确 ('{config_params_from_json.get('dropout')}'). "
                      f"将使用默认值: {[0.3, 0.2, 0.5]}.")
                args_for_run.dropout = [0.3, 0.2, 0.5] # 硬编码默认值以确保安全

        # 其他必要的校验，例如路径是否存在等，可以放在这里或main函数开头
        if args_for_run.load_model_path is None and not args_for_run.train_data:
            print(f"错误: 配置 {run_id_for_logging} 必须提供 --train_data (如果不是加载模型模式)。跳过此配置。")
            continue
        if not args_for_run.test_data:
            print(f"错误: 配置 {run_id_for_logging} 必须提供 --test_data。跳过此配置。")
            continue
        if args_for_run.film_condition_source == 'gene' and args_for_run.load_model_path is None and not args_for_run.train_data:
            print(f"错误: 配置 {run_id_for_logging} 当 film_condition_source='gene' 且非加载模型时，必须提供 --train_data。跳过此配置。")
            continue
        if not args_for_run.output_dir: # 确保基础 output_dir 被指定
            print(f"错误: 配置 {run_id_for_logging} 必须在JSON中指定 'output_dir' (基础输出目录)。跳过此配置。")
            continue


        # 5. 调用 main 函数
        try:
            main(args_for_run)
        except Exception as e:
            print(f"运行配置 {run_id_for_logging} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 6. 资源清理
            print(f"\n清理运行配置 {run_id_for_logging} 的资源...")
            if torch.cuda.is_available():
                print("清空 CUDA 缓存...")
                torch.cuda.empty_cache()
            print("执行垃圾回收...")
            gc.collect()
            plt.close('all') # 关闭所有matplotlib图形
            print(f"配置 {run_id_for_logging} 清理完成。\n")

    print(f"{'='*30} 所有 {len(all_configurations)} 个配置运行完毕 {'='*30}")