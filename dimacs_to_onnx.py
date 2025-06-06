# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:47:49 2024

@author: Edoardo
"""

import csv 
import numpy as np
import torch

def weak_dimacs_parser(filepath):
    info = {}
    cnf = []
    
    with open(filepath, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ')
        for row in csvreader:
            
            # comments, e.g. "c A permuted SAT Competition 2002 formula"
            if row[0] == "c":
                continue
            
            # metadata, e.g. "p cnf 26 70"
            elif row[0] == "p":
                assert(row[1] == "cnf") # can only parse CNF format
                assert(len(row) == 4)
                
                info["vars"] = int(row[2])
                info["clauses"] = int(row[3])
            
            # clauses, e.g. "-25 15 16 0"
            # each entry is an index to the corresponding variable
            # negative number represent negated variables
            # each clause is a disjunction
            else:
                assert(row[-1] == "0") # cannot parse multiline clauses
                
                clause = [int(elem) for elem in row[:-1]]
                cnf.append(clause)
    
    return info, cnf

def cnf_to_nn_params(info, cnf):
    
    n_var = info["vars"]
    n_clause = info["clauses"]
    
    assert(len(cnf) == n_clause)
    
    # encode the disjunctions
    W_or = np.zeros([n_clause, n_var])
    b_or = np.zeros(n_clause)
    
    for i in range(n_clause):
        clause = np.array(cnf[i])
        
        # convert to zero-based indices
        pos_ids = clause[clause > 0] - 1
        neg_ids = -clause[clause < 0] - 1
        
        W_or[i, pos_ids] = -1
        W_or[i, neg_ids] = 1
        b_or[i] = 1 - len(neg_ids)
    
    # encode the binarisation
    W_pass = np.eye(n_var)
    b_pass = np.zeros(n_var)
    
    W_down = 2 * np.eye(n_var)
    b_down = -np.ones(n_var)
    
    # concatenate the first layer parameters
    W_1 = np.vstack([W_or, W_pass, W_down])
    b_1 = np.concatenate([b_or, b_pass, b_down])
    
    # encode the second layer
    n_hidden = len(b_1)
    W_2 = np.zeros([2, n_hidden])
    b_2 = np.zeros(2)
    
    # conjunctions (output >= 1)
    W_2[0, :n_clause] = -1
    b_2[0] = 1
    
    # binarisation (output <= 0)
    W_2[1, n_clause:n_clause+n_var] = 1
    W_2[1, n_clause+n_var:n_hidden] = -1
    
    return (W_1, b_1, W_2, b_2)

def nn_params_to_torch(W_1, b_1, W_2, b_2):
    
    layer_1 = torch.nn.Linear(W_1.shape[1], W_1.shape[0])
    activ_1 = torch.nn.ReLU()
    layer_2 = torch.nn.Linear(W_2.shape[1], W_2.shape[0])
    
    # init layer params
    with torch.no_grad():
        
        layer_1.weight = torch.nn.Parameter(torch.Tensor(W_1))
        layer_1.bias = torch.nn.Parameter(torch.Tensor(b_1))
        
        layer_2.weight = torch.nn.Parameter(torch.Tensor(W_2))
        layer_2.bias = torch.nn.Parameter(torch.Tensor(b_2))
    
    net = torch.nn.Sequential(layer_1, activ_1, layer_2).to("cpu")
    
    return net

def write_torch_to_onnx_file(net, filepath):
    
    # create a dummy input
    # make sure it is two-dimensional, otherwise onnx2c conversion will fail
    # (something to do with MatMul not being fully supported, while GeMM is)
    n = net[0].weight.shape[1]
    x = torch.zeros(1, n)
    
    torch.onnx.export(net, x, filepath, verbose=True)
