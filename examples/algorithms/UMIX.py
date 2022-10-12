import torch
import math
import tqdm
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
import numpy as np
import os
from torch.nn.utils import clip_grad_norm_
from utils import move_to
import math


class UMIX(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        # initialize model
        if config.data_parallel:
            featurizer, classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
            featurizer = featurizer.to(config.device)
            classifier = classifier.to(config.device)
            model = torch.nn.Sequential(featurizer, classifier).to(config.device)
            assert config.device == 'cuda'
            model = torch.nn.DataParallel(model)
        else:
            model = initialize_model(config, d_out).to(config.device)
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        if config.data_parallel:
            self.featurizer = featurizer
            self.classifier = classifier 
        self.seed = config.seed
        self.trajectory_path = os.path.join(config.umix_trajectory_path)

        # hyperparameters
        ## sampling start epoch T_s
        self.T_s = config.umix_T_s
        ## the number of sampling T
        self.T = config.umxi_T
        ## upweight hyperparameter \eta
        self.eta = config.umix_eta
        ## hyperparameter σ to control the probability of doing UMIX
        self.sigma = config.umix_sigma
        ## hyperparameter alpha to control the mixup strength
        self.alpha = config.umix_alpha

        self.trajectory = trajectory(config.train_sample_num, self)
        self.config = config
        self.batch_size = config.batch_size
        self.device = config.device
        self.mixup_type = config.umix_mixup_type

    def update(self, batch, epoch):
        """
        Process the batch, update the log, and update the model
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - outputs (Tensor)
                - y_pred (Tensor)
                - objective (float)
        """
        assert self.is_training
        # process batch
        results = self.process_batch(batch)
        self._update(results)
        # log results        
        self.update_log(results)
        return self.sanitize_dict(results)
        
    def process_batch(self, batch):
        """
        A helper function for update() and evaluate() that processes the batch
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor)
                - g (Tensor)
                - metadata (Tensor)
                - output (Tensor)
                - y_true
        """
        x, y_true, metadata, idx = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        
        p = np.random.rand(1) 
        if self.training:
            if p <= self.sigma:
                weight = move_to(torch.from_numpy(self.trajectory.get_weight(idx)), self.device)
                if self.mixup_type == "vanillamix":
                    x, y_a, y_b, mix, weight_b = mix_up(x, y_true, self.alpha, self.device, weight)
                    outputs = self.model(x)
                elif self.mixup_type == "cutmix":
                    x, y_a, y_b, mix, weight_b = cutmix(x, y_true, self.alpha, self.device, weight)
                    outputs = self.model(x)            
                elif self.mixup_type == "manifoldmix":
                    x = self.featurizer(x)
                    x, y_a, y_b, mix, weight_b = mix_up(x, y_true, self.alpha, self.device, weight)
                    outputs = self.classifier(x)
                results = {
                    'g': g,
                    'y_a': y_a,
                    'y_b': y_b,
                    'y_true': y_true,
                    'y_pred': outputs,
                    'metadata': metadata,
                    'mix':mix,
                    'weight_a':weight,
                    'weight_b':weight_b,
                    'idx': idx
                }
            else:
                weight = move_to(torch.from_numpy(self.trajectory.get_weight(idx)), self.device)
                outputs = self.model(x)
                results = {
                    'g': g,
                    'y_true': y_true,
                    'y_pred': outputs,
                    'metadata': metadata,
                    'idx': idx,
                    'weight': weight
                    }
        elif not self.training:
            outputs = self.model(x)
            results = {
                'g': g,
                'y_true': y_true,
                'y_pred': outputs,
                'metadata': metadata,
                'idx': idx
            }
        return results

    def objective(self, results):
        # compute group losses
        element_wise_losses = self.loss.compute_element_wise(
            results['y_pred'][:self.batch_size],
            results['y_true'][:self.batch_size],
            return_dict=False) 
        loss = element_wise_losses.mean()
        return loss
    
    def _update(self, results):
        """
        Process the batch, update the log, and update the model, group weights, and scheduler.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
                - objective (float)
        """
        if 'mix' in results:
            loss_func = mixup_criterion(results['y_a'], results['y_b'], results['mix'], results['weight_a'],  results['weight_b'])
            objective = loss_func(self.loss.loss_fn, results['y_pred']).mean()
        else:
            element_wise_losses = self.loss.compute_element_wise(results['y_pred'], results['y_true'], return_dict=False)
            objective = (element_wise_losses * results['weight']).mean()
        
        self.model.zero_grad()
        objective.backward()
        if self.max_grad_norm:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.step_schedulers(
            is_epoch=False,
            metrics=results,
            log_access=False)
        results['objective'] = objective.item()


class trajectory(object):
    def __init__(self, n_data, args):
        
        self.trajectoris = np.load(os.path.join(args.trajectory_path, "trajectory_"+str(args.seed)+".npy"), allow_pickle=True)
        self.n_data = n_data
        self.T_s = args.T_s
        self.T = args.T
        self.eta = args.eta

    def get_weight(self, idx):
        weight = (1-np.mean(self.trajectoris[idx, self.T_s : self.T_s + self.T], axis=1)) * self.eta + 1
        return weight


def mix_up(x, l, beta_param, device,  weight):
    """
    Args:
        x: the input image batch [batch_size, H, W, C]
        l: the label batch  [batch_size, num_of_class]
        v: mentornet weights
        beta_param: the parameter to sample the weight average.
    Returns:
        result: The mixed images and label batches.
    """
    batch_size = x.shape[0]
    idx = move_to(torch.randperm(batch_size), device)
    x_b = x[idx]
    l_b = l[idx]
    lam = np.random.beta(beta_param, beta_param)
    xmix = lam * x + (1 - lam) * x_b
    y_a = l
    y_b = l_b
    weight_b = weight[idx]
    return xmix, y_a, y_b, lam, weight_b


def mixup_criterion(y_a, y_b, lam, weight_a, weight_b):
    return lambda criterion, pred: lam * weight_a * criterion(pred, y_a) +  (1 - lam) * weight_b * criterion(pred, y_b)

def cutmix(x, l, beta_param, device, weight):
    batch_size = x.shape[0]
    idx = move_to(torch.randperm(batch_size), device)
    l2 = l[idx]
    weight2 = weight[idx]
    lam = np.random.beta(beta_param, beta_param)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, l, l2, lam, weight2

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
