# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:06:59 2024

@author: Edoardo
"""

import csv
import random
import numpy as np

def print_dimacs(filepath, info, cnf, verdict="Unsat"):
    
    with open(filepath, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=' ')
        
        # print comment
        csvwriter.writerow(["c", "NeuroCodeBench", "2.0",
                            "CNF", "formula", "with", "verdict", verdict])
        
        # print info
        csvwriter.writerow(["p", "cnf", str(info["vars"]), str(info["clauses"])])
        
        # print clauses
        for clause in cnf:
            clause_list = [str(lit) for lit in clause] + ["0"]
            csvwriter.writerow(clause_list)

def generate_sat(n_var, n_clause, max_fail = 100):
    
    cnf = []
    n_fail = 0
    
    truth = np.random.rand(n_var) < 0.5
    while len(cnf) < n_clause and n_fail < max_fail:
        
        v = np.random.choice(n_var, size=np.random.randint(0, n_var), replace=False)
        t = np.random.rand(len(v)) < 0.5
        
        if (truth[v] == t).any():
            
            # convert to one-based indices
            pos_v = v[t] + 1
            neg_v = -v[np.logical_not(t)] - 1
            
            cnf.append(list(np.concatenate([pos_v, neg_v])))
            n_fail = 0
        
        else:
            n_fail = n_fail+ 1
            
    # eliminate repeated clauses
    cnf = list(set(tuple(clause) for clause in cnf))
    info = {"vars": n_var, "clauses": len(cnf)}
    
    return info, cnf

def generate_unsat(n_var, n_clause, max_fail = 100):
    
    cnf = []
    n_fail = 0
    
    assert(n_var >= 1 and n_clause >= 2)
    
    # start with a simple contradiction (x1) and (!x1)
    cnf.append([+1])
    cnf.append([-1])
    
    while len(cnf) < n_clause and n_fail < max_fail:
        
        # randomly choose an existing clause
        # prioritise shorter clauses
        freq = 1 / np.array([len(clause) for clause in cnf])
        i = int(np.random.choice(len(cnf), size=1, p=freq / np.sum(freq))[0])
        clause = cnf[i]
        
        # list the variables not in the clause
        in_set = {abs(v) for v in clause}
        out_set = [j + 1 for j in range(n_var) if j + 1 not in in_set]
        
        # duplicate the clause by adding the new (negated) variable
        if len(out_set) > 0:
            new_var = random.choice(out_set)
            cnf[i] = clause + [new_var]
            cnf.append(clause + [-new_var])
            n_fail = 0
        
        else:
            n_fail = n_fail+ 1
    
    # eliminate repeated clauses
    cnf = list(set(tuple(clause) for clause in cnf))
    info = {"vars": n_var, "clauses": len(cnf)}
    
    return info, cnf
