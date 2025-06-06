# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 18:35:53 2025

@author: Edoardo
"""

def vnnlib_lines(n_var, verdict):
    lines = []
    
    # intro comment:
    # NeuroCodeBench 2.0: SAT ReLU instance with 12 variables and SAT verdict
    lines.append("; NeuroCodeBench 2.0: SAT ReLU instance with " +
                 str(n_var) + " variables and " + verdict + " verdict")
    lines.append("")
    
    # input variables:
    # (declare-const X_0 Real)
    lines.append("; Input Variables")
    for i in range(n_var):
        lines.append("(declare-const X_" + str(i) + " Real)")
    lines.append("")
    
    # output variables
    lines.append("; Output Variables")
    lines.append("(declare-const Y_0 Real)")
    lines.append("(declare-const Y_1 Real)")
    lines.append("")
    
    # input constraints
    # (assert (<= X_54 1.0))
    # (assert (>= X_54 0.0))
    lines.append("; Input Constraints")
    for i in range(n_var):
        lines.append("(assert (<= X_" + str(i) + " 1.0))")
        lines.append("(assert (>= X_" + str(i) + " 0.0))")
    lines.append("")
    
    # output constraints
    lines.append("; Output Constraints")
    lines.append("(assert (>= Y_0 1.0))")
    lines.append("(assert (<= Y_1 0.0))")
    
    return lines
