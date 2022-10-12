import torch
import math
import tqdm
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
import numpy as np
import os
from torch.nn.utils import clip_grad_norm_
from utils import move_to

class UMIX_trajectory(SingleModelAlgorithm):
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

        self.n_epochs = config.n_epochs
        self.seed = config.seed
        self.log_dir = config.log_dir

        self.n_train_steps = n_train_steps
        self.trajectory = trajectory(config.train_sample_num, self)
                
        self.config = config
        self.batch_size = config.batch_size
        self.device = config.device
        self.current_epoch = 0
        
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
        element_wise_losses = self.loss.compute_element_wise(
            results['y_pred'][:self.batch_size],
            results['y_true'][:self.batch_size],
            return_dict=False) 
        loss = element_wise_losses.mean()
        return loss

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
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self.trajectory.save_trajectoris()

        results = self.process_batch(batch)
        self._update(results)
        # log results        
        self.update_log(results)
        return self.sanitize_dict(results)
    
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
        element_wise_losses = self.loss.compute_element_wise(results['y_pred'], results['y_true'], return_dict=False)
        objective  = element_wise_losses.mean()
        _, predicted = torch.max(results['y_pred'].data, 1)
        correctness = predicted.eq(results['y_true'].data).cpu().numpy()
        self.trajectory.trajectoris_update(results['idx'].cpu().numpy(), self.current_epoch, correctness)
        
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
        self.trajectoris = np.zeros((n_data, args.n_epochs))
        self.n_data = n_data
        self.log_dir = args.log_dir
        self.seed = args.seed

    def trajectoris_update(self, data_idx, epoch, correctness):
        self.trajectoris[data_idx, epoch] = correctness

    def save_trajectoris(self):
        os.makedirs(os.path.join(self.log_dir, 'trajectory'), exist_ok=True)
        np.save(os.path.join(self.log_dir, 'trajectory', 'trajectory_'+str(self.seed)+'.npy'), self.trajectoris)