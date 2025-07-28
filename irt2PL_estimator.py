"""
pytorch 2PL IRT Model Estimator

This module provides an implementation of 2-Parameter Logistic (2PL) IRT model
estimation using the Expectation-Maximization (EM) algorithm. 
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict, Optional
import warnings


class Simple2PL(nn.Module):
    """
    Standard 2-Parameter Logistic IRT model.
    
    This model estimates examinee ability (theta) and item parameters
    (difficulty beta and discrimination alpha) using the classic 2PL formulation:
    P(X_ij = 1) = sigmoid(alpha_j * (theta_i - beta_j))
    """
    
    def __init__(self, n_examinees: int, n_items: int, model_type: str = "2PL"):
        """
        Initialize the 2PL model.
        
        Args:
            n_examinees (int): Number of examinees in the dataset
            n_items (int): Number of unique items in the dataset
            model_type (str): Either "1PL" (Rasch) or "2PL" 
        """
        super().__init__()
        
        # Store model configuration
        self.n_examinees = n_examinees
        self.n_items = n_items
        self.model_type = model_type
        
        # Examinee ability parameters (theta)
        self.theta = nn.Parameter(torch.randn(self.n_examinees))
        
        # Item difficulty parameters (beta) - initialized near zero
        self.beta = nn.Parameter(torch.randn(self.n_items) * 0.1)
        
        # Item discrimination parameters (alpha) - initialized near 1
        self.alpha = nn.Parameter(torch.ones(self.n_items) + torch.randn(self.n_items) * 0.1)
        
        # For 1PL (Rasch) model, fix discrimination parameters to 1
        if self.model_type == "1PL":
            self.alpha.requires_grad = False
        
        # Sigmoid activation for probabilities
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, examinee_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute probability of correct response.
        
        Args:
            examinee_ids (tensor): Tensor of examinee indices
            item_ids (tensor): Tensor of item indices
            
        Returns:
            tensor: Probabilities of correct response for each examinee-item pair
        """
        # Extract relevant parameters for this batch
        theta_batch = self.theta[examinee_ids]  # Abilities for these examinees
        alpha_batch = self.alpha[item_ids]      # Discriminations for these items
        beta_batch = self.beta[item_ids]        # Difficulties for these items
        
        # Apply identifiability constraints
        theta_centered = theta_batch - self.theta.mean()  # Center abilities
        if self.model_type == "2PL":
            theta_scaled = theta_centered / self.theta.std()  # Scale to unit variance
        else:
            theta_scaled = theta_centered
        
        beta_centered = beta_batch - self.beta.mean()  # Center difficulties
        
        # Compute 2PL probability: P = sigmoid(alpha * (theta - beta))
        logit = alpha_batch * (theta_scaled - beta_centered)
        prob = self.sigmoid(logit)
        
        return prob


class Simple2PLDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for 2PL IRT estimation.
    
    Expects DataFrame with columns: examinee_id, item_id, response
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize dataset from DataFrame.
        
        Args:
            df (DataFrame): Input data with columns [examinee_id, item_id, response]
        """
        # Validate input format
        required_cols = ['examinee_id', 'item_id', 'response']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create mappings from original IDs to sequential indices
        self.examinees = sorted(df['examinee_id'].unique())
        self.items = sorted(df['item_id'].unique())
        
        self.examinee_to_idx = {ex: idx for idx, ex in enumerate(self.examinees)}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.items)}
        
        # Convert to indexed format
        df_indexed = df.copy()
        df_indexed['examinee_idx'] = df_indexed['examinee_id'].map(self.examinee_to_idx)
        df_indexed['item_idx'] = df_indexed['item_id'].map(self.item_to_idx)
        
        # Store processed data
        self.data = df_indexed[['examinee_idx', 'item_idx', 'response']].values
        self.n_examinees = len(self.examinees)
        self.n_items = len(self.items)
        
    def __len__(self) -> int:
        """Return number of response records."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single response record.
        
        Args:
            idx: Index of the record to retrieve
            
        Returns:
            tuple: (features, response) where features contains [examinee_idx, item_idx]
        """
        record = self.data[idx]
        features = torch.tensor(record[:2], dtype=torch.long)  # examinee_idx, item_idx
        response = torch.tensor(record[2], dtype=torch.float)  # response
        return features, response


def em_step(optimizer: torch.optim.Optimizer, 
           loader: torch.utils.data.DataLoader,
           model: Simple2PL,
           loss_fn: nn.Module,
           device: torch.device,
           max_iter: int = 100,
           eps: float = 1e-4,
           verbose: bool = False) -> Tuple[int, float, list]:
    """
    Perform one step of the EM algorithm.
    
    Args:
        optimizer: PyTorch optimizer for parameter updates
        loader: DataLoader containing training data
        model: The 2PL model to train
        loss_fn: Loss function (typically BCELoss)
        device: Computing device (CPU or CUDA)
        max_iter: Maximum number of iterations within this step
        eps: Convergence threshold for loss change
        verbose: Whether to print detailed iteration info
        
    Returns:
        tuple: (iterations_run, final_loss, loss_history)
    """
    model.train()
    loss_log = []
    prev_loss = float('inf')
    
    for iteration in tqdm(range(max_iter), desc='EM Step', disable=not verbose):
        epoch_loss = 0.0
        n_samples = 0
        
        # Process all batches
        for features, responses in loader:
            features, responses = features.to(device), responses.to(device)
            
            # Extract examinee and item indices
            examinee_ids = features[:, 0]
            item_ids = features[:, 1]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predicted_probs = model(examinee_ids, item_ids)
            
            # Compute loss
            loss = loss_fn(predicted_probs, responses)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item() * len(features)
            n_samples += len(features)
        
        # Compute average loss
        avg_loss = epoch_loss / n_samples
        loss_log.append(avg_loss)
        
        # Check convergence
        if iteration > 0 and abs(prev_loss - avg_loss) < eps:
            if verbose:
                print(f"    Converged at iteration {iteration} (loss change: {abs(prev_loss - avg_loss):.6f})")
            break
        
        prev_loss = avg_loss
        
        if verbose and iteration % 10 == 0:
            print(f"    Iteration {iteration}: Loss = {avg_loss:.6f}")
    
    return iteration, avg_loss, loss_log


def estimate_2pl_parameters(df: pd.DataFrame,
                          model_type: str = "2PL",
                          max_epochs: int = 50,
                          max_iter_per_step: int = 100,
                          batch_size: int = 512,
                          lr_theta: float = 0.1,
                          lr_items: float = 0.01,
                          convergence_threshold: float = 1e-4,
                          verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Estimate 2PL IRT parameters using EM algorithm.
    
    Args:
        df (DataFrame): Input data with columns [examinee_id, item_id, response]
        model_type (str): Either "1PL" (Rasch) or "2PL"
        max_epochs (int): Maximum number of EM epochs
        max_iter_per_step (int): Maximum iterations within each E/M step
        batch_size (int): Batch size for training
        lr_theta (float): Learning rate for ability parameters (E-step)
        lr_items (float): Learning rate for item parameters (M-step)
        convergence_threshold (float): Convergence threshold for EM algorithm
        verbose (bool): Whether to print training progress
        
    Returns:
        dict: Dictionary containing estimated parameters:
              - 'theta': examinee abilities
              - 'alpha': item discriminations
              - 'beta': item difficulties
              - 'examinees': original examinee IDs
              - 'items': original item IDs
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # Create dataset and data loader
    dataset = Simple2PLDataset(df)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = Simple2PL(dataset.n_examinees, dataset.n_items, model_type).to(device)
    loss_fn = nn.BCELoss()
    
    # Separate optimizers for E-step (theta) and M-step (item parameters)
    theta_optimizer = torch.optim.AdamW([model.theta], lr=lr_theta)
    item_optimizer = torch.optim.AdamW([model.alpha, model.beta], lr=lr_items)
    
    # EM algorithm
    prev_loss = float('inf')
    all_losses = []
    
    if verbose:
        print(f"\nFitting {model_type} model to {len(df)} responses")
        print(f"Examinees: {dataset.n_examinees}, Items: {dataset.n_items}")
        print("-" * 50)
    
    for epoch in range(max_epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{max_epochs}")
        
        # E-step: Update ability parameters (theta)
        e_iter, e_loss, e_log = em_step(
            theta_optimizer, loader, model, loss_fn, device,
            max_iter_per_step, convergence_threshold, verbose=False
        )
        if verbose:
            print(f"  E-step: {e_iter} iterations, Loss = {e_loss:.6f}")
        
        # M-step: Update item parameters (alpha, beta)
        m_iter, m_loss, m_log = em_step(
            item_optimizer, loader, model, loss_fn, device,
            max_iter_per_step, convergence_threshold, verbose=False
        )
        if verbose:
            print(f"  M-step: {m_iter} iterations, Loss = {m_loss:.6f}")
        
        all_losses.extend(e_log + m_log)
        
        # Check overall convergence
        if abs(prev_loss - m_loss) < convergence_threshold:
            if verbose:
                print(f"\nEM algorithm converged at epoch {epoch + 1}")
            break
        
        prev_loss = m_loss
    
    # Extract final parameters
    model.eval()
    with torch.no_grad():
        # Apply identifiability constraints for output
        theta_final = model.theta.cpu().numpy()
        alpha_final = model.alpha.cpu().numpy()
        beta_final = model.beta.cpu().numpy()
        
        # Center and scale appropriately
        theta_final = theta_final - theta_final.mean()
        if model_type == "2PL":
            theta_final = theta_final / theta_final.std()
        
        beta_final = beta_final - beta_final.mean()
    
    if verbose:
        print(f"\nEstimation complete!")
        print(f"Final loss: {m_loss:.6f}")
        print(f"Theta range: [{theta_final.min():.3f}, {theta_final.max():.3f}]")
        print(f"Alpha range: [{alpha_final.min():.3f}, {alpha_final.max():.3f}]")
        print(f"Beta range: [{beta_final.min():.3f}, {beta_final.max():.3f}]")
    
    return {
        'theta': theta_final,
        'alpha': alpha_final,
        'beta': beta_final,
        'examinees': dataset.examinees,
        'items': dataset.items,
        'final_loss': m_loss,
        'loss_history': all_losses
    }


def calculate_2pl_irt_discriminant(df: pd.DataFrame, 
                                 model_type: str = "2PL",
                                 verbose: bool = False) -> np.ndarray:
    """
    Extract item discrimination parameters from 2PL IRT model.
    
    This is a simplified interface that focuses specifically on extracting
    discrimination parameters, similar to the function in old_2pl_code.py
    but using the correct EM algorithm approach.
    
    Args:
        df (DataFrame): Input data with columns [examinee_id, item_id, response]
        model_type (str): Either "1PL" (Rasch) or "2PL"
        verbose (bool): Whether to print estimation progress
        
    Returns:
        ndarray: Array of discrimination parameters for each item
    """
    # Estimate all parameters
    results = estimate_2pl_parameters(df, model_type=model_type, verbose=verbose)
    
    # Return only discrimination parameters
    return results['alpha']