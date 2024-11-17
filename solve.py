import argparse as ap
import numpy as np
import os
from matrices import FormalisationObjects, FormalisationMatrix
from files import output_file, limit_output
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math

import juliapkg
juliapkg.require_julia("=1.10.3")
juliapkg.resolve()
from juliacall import Main as jl

def shutdown_julia():
    """
    This method manually clears Julia memory by seting all values to None for garbage collection
    """
    try:
        jl.seval("empty!(names(Main, all=true))")
        #print("julia has been reset.")
    except Exception as e:
        pass
        #print("error shutting down julia "+str(e))

np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=4)

def aggregate(P_list, 
    J_list, 
    w, 
    country_dict, 
    PP_list, 
    PJ_list, 
    Pw, 
    Pcountry_dict, 
    principle_data, 
    limit_p_filename):
    """
    This function is used by the experiment_runner.py file for aggregation
    """

    ############################################
    # Aggregate Principles and find HCVA point #
    ############################################
    A, b = FormalisationMatrix(PP_list, PJ_list, Pw, 1, True)
    cons_1, _, ua = L1(A, b)
    A, b = FormalisationMatrix(PP_list, PJ_list, Pw, np.inf, True)
    cons_l, _, _, = Linf(A, b)
    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    p = 1
    #print('{:.2f} \t \t {:.4f}'.format(p, ua))
    incr = 0.1
    p_list = [1.0]
    u_list = [ua]
    cons_list = [cons_1]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]

    while p < 10:
        p += incr
        A, b = FormalisationMatrix(PP_list, PJ_list, Pw, p, True)
        cons, _, ub = Lp(A, b, p)
        p_list.append(p)
        u_list.append(ub)
        cons_list.append(cons)
        dist_1p = np.linalg.norm(cons_1 - cons, p)
        dist_pl = np.linalg.norm(cons_l - cons, p)
        dist_1p_list.append(dist_1p)
        dist_pl_list.append(dist_pl)
        #print('{:.2f} \t \t {:.4f}'.format(p, ub))

    # Finding cutoff point to stop bias to egalitarianism
    cut_point = 10
    incr = 0.1
    j = 0
    for i in np.arange(1 + incr, 10, incr):
        cons = cons_list[j]
        dist_1p = np.linalg.norm(cons_1 - cons, i)
        dist_pl = np.linalg.norm(cons_l - cons, i)
        j += 1
        # Note: Hard Coded \epsilon value
        if (abs(dist_1p - dist_pl) < 0.005):
            cut_point = i
            #print('Not improving anymore, stopping!')
            break

    cut_list = [cons_list[i] for i in range(len(cons_list)) if p_list[i] <= cut_point]
    #print("DEBUG: cut_list length is: ", len(cut_list))
    con_vals = [0, 0]
    for j in range(2):
        con_vals[j] = sum(i[j+1] for i in cut_list) / len(cut_list)

    hcva_point = 1.0 
    best_dist = 999
    for j in range(len(cons_list)):
        dist = [abs(cons_list[j][1] - con_vals[0]), abs(cons_list[j][2] - con_vals[1])]
        dist = sum(dist)
        if dist < best_dist:
            best_dist = dist
            # to convert from ordinal list num to corresponding p
            hcva_point = (j/10)+1


    # Note because we are not interested in storing all value systems for HCVA we can re-use variable names for consistency

    ###################################################
    # Aggregate Preferences and find transition point #
    ###################################################

    A, b = FormalisationMatrix(P_list, J_list, w, 1, True)
    cons_1, _, ua = L1(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, True)
    cons_l, _, _, = Linf(A, b)
    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    p = 1
    #print('{:.2f} \t \t {:.4f}'.format(p, ua))
    incr = 0.1
    p_list = [1.0]
    u_list = [ua]
    cons_list = [cons_1]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]
    diff = np.inf

    while p < 10:
        p += incr
        A, b = FormalisationMatrix(P_list, J_list, w, p, True)
        cons, _, ub = Lp(A, b, p)
        p_list.append(p)
        u_list.append(ub)
        cons_list.append(cons)
        dist_1p = np.linalg.norm(cons_1 - cons, p)
        dist_pl = np.linalg.norm(cons_l - cons, p)
        dist_1p_list.append(dist_1p)
        dist_pl_list.append(dist_pl)
        #print('{:.2f} \t \t {:.4f}'.format(p, ub))
        # Calculating transition point
        if abs(dist_1p - dist_pl) < diff:
                diff = abs(dist_1p - dist_pl)
                t_point = p
    # store all cons_vals
    preference_cons_vals = cons_list
    
    #####################
    # Aggregate actions #
    #####################

    A, b = FormalisationMatrix(P_list, J_list, w, 1, False)
    cons_1, _, ua = L1(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, False)
    cons_l, _, _, = Linf(A, b)
    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    p = 1
    #print('{:.2f} \t \t {:.4f}'.format(p, ua))
    incr = 0.1
    p_list = [1.0]
    u_list = [ua]
    cons_list = [cons_1]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]
    while p < 10:
        p += incr
        A, b = FormalisationMatrix(P_list, J_list, w, p, False)
        cons, _, ub = Lp(A, b, p)
        p_list.append(p)
        u_list.append(ub)
        cons_list.append(cons)
        dist_1p = np.linalg.norm(cons_1 - cons, p)
        dist_pl = np.linalg.norm(cons_l - cons, p)
        dist_1p_list.append(dist_1p)
        dist_pl_list.append(dist_pl)
    action_cons_vals = cons_list

    return t_point, hcva_point, preference_cons_vals, action_cons_vals

def aggregate_all_p(P_list, J_list, w):
    """
    Used by pv argument to aggregate on principles in solve.py
    """
    A, b = FormalisationMatrix(P_list, J_list, w, 1, True)
    cons_1, _, ua = L1(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, True)
    cons_l, _, _, = Linf(A, b)
    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    p = 1
    #print('{:.2f} \t \t {:.4f}'.format(p, ua))
    incr = 0.1
    p_list = [1.0]
    u_list = [ua]
    cons_list = [cons_1]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]

    while p < 10:
        p += incr
        A, b = FormalisationMatrix(P_list, J_list, w, p, True)
        cons, _, ub = Lp(A, b, p)
        p_list.append(p)
        u_list.append(ub)
        cons_list.append(cons)
        dist_1p = np.linalg.norm(cons_1 - cons, p)
        dist_pl = np.linalg.norm(cons_l - cons, p)
        dist_1p_list.append(dist_1p)
        dist_pl_list.append(dist_pl)
        #print('{:.2f} \t \t {:.4f}'.format(p, ub))
    return p_list, u_list, cons_list, dist_1p_list, dist_pl_list, cons_1, cons_l

def print_consensus(cons):
    print('Rs =')
    if args.v:
        print(cons.reshape((m, m)))
    else:
        print(cons.reshape((2 * m, m)))

def make_decision(cons_prefs, cons_actions) -> str:
    """
    This function computes a simple decision based on the consensus value system
    $(P[v_1,v_2]*a_{v_1}(i))+(P[v_2,v_1]*a_{v_2}(i))$

    cons_prefs in format [rel-rel, rel-nonrel, nonrel-rel,nonrel-nonrel]
    cons_actions in format [rel_adp_p, rel_div_p, nonrel_adp_p, nonrel_div_p]
    """
    # adp can be between -1 and 1, as cons_prefs can be between 0 and 1, and cons_actions can be between -1 and 1
    adp = (cons_prefs[1] * cons_actions[0]) + (cons_prefs[2] * cons_actions[2])
    div = (cons_prefs[1] * cons_actions[1]) + (cons_prefs[2] * cons_actions[3])
    decision = [adp, div]
    return decision

def aggregate_principles(aggregation_type, filename, con_p=0.0, 
                     P_list=None, 
                     J_list=None, 
                     w=None,
                     principle_val=None):
    """
    We run aggregation of the action value matrices and store in a file
    Used in the solve.py file for argument -pv (principle value aggregation)
    """
    consensus_vals = []

    ## The same as in elif args.g
    # Doing compute on 1 and np.inf allows for distance calc for P
    A, b = FormalisationMatrix(P_list, J_list, w, 1, aggregation_type)
    cons_1, _, ua = L1(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, aggregation_type)
    cons_l, _, _, = Linf(A, b)
    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    
    p = principle_val    
    p_list = []
    u_list = []
    cons_list = []
    dist_1p_list = []
    dist_pl_list = []

    A, b = FormalisationMatrix(P_list, J_list, w, p, aggregation_type)
    cons, _, ub = Lp(A, b, p)
    p_list.append(p)
    u_list.append(ub)
    cons_list.append(cons)
    dist_1p = np.linalg.norm(cons_1 - cons, p)
    dist_pl = np.linalg.norm(cons_l - cons, p)
    dist_1p_list.append(dist_1p)
    dist_pl_list.append(dist_pl)
    #print('{:.2f} \t \t {:.4f}'.format(p, ub))
    consensus_vals = [p, ub, cons, dist_1p, dist_pl]
    
    """
    print("DEBUG: Saving to file, ", filename, " and returning cons list")
    output_file(
        p_list,
        u_list,
        cons_list,
        dist_1p_list,
        dist_pl_list,
        aggregation_type,
        filename)
    """
    
    # Cut off n values, only interested in p values
    # Note: n values (in action judgements) are the opposite of the p values
    cons_list = cons_list[0][:4]
    return cons_list


def L1(A, b):
    """
    This function runs the L1 norm on values and returns consensus.
    Note that this is the fully utilitarian case P=1
    """
    import cvxpy as cp
    # create variables
    l = A.shape[1]
    t = cp.Variable(len(b), integer=False)
    x = cp.Variable(l, integer=False)
    # create constraints
    constraint1 = [A @ x - b >= -t]
    constraint2 = [A @ x - b <= t]
    constraints = constraint1 + constraint2
    cost = cp.sum(t)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    # optimize model
    prob.solve(solver='ECOS', verbose=False)
    # prob.solve(solver='GLPK', verbose=True)
    cons = list(x.value)
    cons = np.array(cons)
    obj = prob.value
    #print("obj value:", obj)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, 1)

def L2(A, b):
    """
    This function runs the L2 norm on values and returns consensus
    P=2
    """
    cons, res, rank, a = np.linalg.lstsq(A, b, rcond=None)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r)


def Linf(A, b):
    """
    This function runs the Linf norm on values and returns consensus
    Note that this is the fully egalitarian case P=inf
    """
    import cvxpy as cp
    # create variables
    l = A.shape[1]
    t = cp.Variable(1, integer=False)
    x = cp.Variable(l, integer=False)
    # create constraints
    constraint1 = [A @ x - b >= -t * np.ones_like(b)]
    constraint2 = [A @ x - b <= t * np.ones_like(b)]
    constraints = constraint1 + constraint2
    prob = cp.Problem(cp.Minimize(t), constraints)
    # optimize model
    prob.solve(solver='ECOS', verbose=False)
    # prob.solve(solver='GLPK', verbose=True)
    cons = list(x.value)
    cons = np.array(cons)
    obj = prob.value
    #print("obj value: ", obj)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, np.inf)

def IRLS(A, b, p, max_iter=int(1e6), e=1e-3, d=1e-4):
    """
    This function runs the IRLS method for finding consensus for any P >= 3
    using a python implementation
    """
    # l = A.shape[1]
    n = A.shape[0]
    D = np.repeat(d, n)
    W = np.diag(np.repeat(1, n))
    x = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ b  # initial LS solution
    for i in range(max_iter):
        W_ = np.diag(np.power(np.maximum(np.abs(b - A @ x), D), p - 2))
        # reweighted LS solution
        x_ = np.linalg.inv(A.T @ W_ @ A) @ A.T @ W_ @ b
        e_ = sum(abs(x - x_))
        if e_ < e:
            break
        else:
            W = W_
            x = x_
    r = np.abs(A @ x - b)
    return x, r, np.linalg.norm(r, p)

def Lp(A, b, p):
    # l = A.shape[1]
    if p >= 2 :  # pIRLS implementation (NIPS 2019) (always use this for continuity)
        jl.include(os.path.dirname(
                os.path.realpath(__file__)) +
            '/IRLS-pNorm.jl')
        # constraints needed for pIRLS (empty)
        C = np.zeros_like(A)
        d = np.zeros_like(b)
        epsilon = 1e-10
        cons, it = jl.pNorm(epsilon, A, b.reshape(-1, 1),
                              p, C, d.reshape(-1, 1))
        # cons, it = IRLS.pNorm(epsilon, A, b.reshape(-1, 1), p, C, d.reshape(-1, 1))
        r = np.abs(A @ cons - b)
        jl.collector()
        return cons, r, np.linalg.norm(r, p)
    else:  # vanilla IRLS implementation
        return IRLS(A, b, p)

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-n', type=int, default=7, help='n')
    parser.add_argument('-m', type=int, default=2, help='m')
    parser.add_argument('-p', type=float, default=10, help='p')
    parser.add_argument('-e', type=float, default=1e-4, help='e')
    parser.add_argument(
        '-f',
        type=str,
        default="/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/processed_data_one_action_ess.csv_with_factor_2.5_5.0.csv",
        #default='/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/form_data.csv',
        help='CSV file with personal data')
    parser.add_argument(
        '-w',
        type=int,
        default=0,
        help='1 people participated in the study and 2 population of each country')
    parser.add_argument(
        '-i',
        type=str,
        help='computes equivalent p given an input consensus')
    parser.add_argument('-o', type=str, help='write consensus to file', default="consensus_part2.csv")
    parser.add_argument(
        '-v',
        help='computes the preference aggregation',
        default=False,
        action='store_true')
    parser.add_argument('-l', help='compute the limit p', action='store_true', default=False)
    parser.add_argument(
        '-t',
        help='compute the threshold p',
        action='store_true',
        default=False
        )
    parser.add_argument(
        '-g',
        type=str,
        #default='14-11-results-factor-2.5-5.0-prefs.csv',
        #default='14-11-results-factor-2.5-5.0.csv',
        default='none',
        help='store results in csv')
    
    parser.add_argument(
        '-pf',
        type=str,
        #default=None,
        default='/home/ia23938/Documents/GitHub/ValueSystemsAggregation/process_data/14-11-processed_data_with_principles_ess.csv',
        #default='/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/form_principles.csv',
        help='CSV file with principle data'
    )    
    parser.add_argument(
        '-pv',
        type=bool,
        #default=False, 
        default=True,
        help='Compute the P value consensus aggregation method'
    )
    parser.add_argument(
        '-ex',
        type=str,
        default="",
        help="Generate explanation of agent with index as arg"
    )

    args = parser.parse_args()

    p = args.p
    n = args.n
    m = args.m

    # init for personal aggregation

    P_list, J_list, w, country_dict = FormalisationObjects(
        filename=args.f, delimiter=',', weights=args.w)
    
    if args.pv:
        PP_list, PJ_list, Pw, Pcountry_dict = FormalisationObjects(
            filename=args.pf, delimiter=',', weights=args.w)

    # Compute the limit P
    if args.l:
        print("limit time!")
        A, b = FormalisationMatrix(P_list, J_list, w, 1, args.v)
        cons_1, _, _, = L1(A, b)
        print('L1 =', cons_1)
        A, b = FormalisationMatrix(P_list, J_list, w, np.inf, args.v)
        cons_inf, _, _, = Linf(A, b)
        dist_1inf = np.linalg.norm(cons_1 - cons_inf, 1)
        print('Dist1 = {:.4f}'.format(dist_1inf))
        p = 1
        incr = 0.1
        du_o = 1.0
        while p < 100:
            p += incr
            A, b = FormalisationMatrix(P_list, J_list, w, p, args.v)
            cons, _, _, = Lp(A, b, p)
            dist_p = np.linalg.norm(cons - cons_inf, p)
            du = dist_p / dist_1inf
            print('U{:.2f} = {:.4f}'.format(p, du))
            if du < args.e:
                print(' (ΔU/U1 = {:.4f} < {})'.format(du, args.e))
                break
            else:
                du_o = du
                print(' (ΔU/U1 = {:.4f} > {})'.format(du, args.e))

    # Compute personal aggregation only and store in .csv 
    elif args.g != 'none':
        A, b = FormalisationMatrix(P_list, J_list, w, 1, args.v)
        cons_1, _, ua = Lp(A, b, 1)
        print('L1 =', cons_1)
        A, b = FormalisationMatrix(P_list, J_list, w, np.inf, args.v)
        cons_l, _, _, = Linf(A, b)
        dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
        dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
        p = 1
        print('{:.2f} \t \t {:.4f}'.format(p, ua))
        incr = 0.1
        p_list = [1.0]
        u_list = [ua]
        cons_list = [cons_1]
        dist_1p_list = [dist_1p]
        dist_pl_list = [dist_pl]

        while p < args.p:
            p += incr
            A, b = FormalisationMatrix(P_list, J_list, w, p, args.v)
            cons, _, ub = Lp(A, b, p)
            p_list.append(p)
            u_list.append(ub)
            cons_list.append(cons)
            dist_1p = np.linalg.norm(cons_1 - cons, p)
            dist_pl = np.linalg.norm(cons_l - cons, p)
            dist_1p_list.append(dist_1p)
            dist_pl_list.append(dist_pl)
            #print('{:.2f} \t \t {:.4f}'.format(p, ub))
            print('p: {:.2f}, cons: '.format(p), cons)

        output_file(
            p_list,
            u_list,
            cons_list,
            dist_1p_list,
            dist_pl_list,
            args.v,
            args.g)
        # simple_output_file(p_list, u_list, 'copy' + args.g)
        # simple_output_file(p_list, dist_1p_list, 'dist_1' + args.g)
        # simple_output_file(p_list, dist_pl_list, 'dist_inf' + args.g)

    # Comptue the threshold P, and print transition point
    elif args.t:
        print("Threshold time!")
        A, b = FormalisationMatrix(P_list, J_list, w, 1, args.v)
        cons_1, r_1, u_1 = L1(A, b)
        print("cons_1", cons_1)
        #cons_1 = [-0.027939325961567275,0.013170938317774624,0.027939325961567275,-0.013170938317774624]
        A, b = FormalisationMatrix(P_list, J_list, w, np.inf, args.v)
        cons_l, r_l, u_l = Linf(A, b)
        #cons_l = [-0.052294976814419566,0.01052405695704031,0.052294976814419566,-0.01052405695704031]
        diff = np.inf
        incr = 0.1

        p_list = []
        dist_p_list = []
        dist_inf_list = []
        diff_list = []

        for i in np.arange(1 + incr, p, incr):
            A, b = FormalisationMatrix(P_list, J_list, w, i, args.v)
            cons, r, u = Lp(A, b, i)
            dist_1p = np.linalg.norm(cons_1 - cons, i)
            dist_pl = np.linalg.norm(cons_l - cons, i)
            if (abs(dist_1p - dist_pl) < args.e):
                best_p = i
                print('Not improving anymore, stopping!')
                #break
            else:
                print('p = {:.2f}'.format(i))
                print('Distance L1<-->L{:.2f} = {:.4f}'.format(i, dist_1p))
                print(
                    'Distance L{:.2f}<-->L{:.2f} = {:.4f}'.format(i, p, dist_pl))
                print(
                    'Difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(
                        i, i, p, abs(
                            dist_1p - dist_pl)))
                print(
                    'Current best difference (L1<-->L{:.2f}) - (L{:.2f}<-->L{:.2f}) = {:.4f}'.format(i, i, p, diff))
                if abs(dist_1p - dist_pl) < diff:
                    diff = abs(dist_1p - dist_pl)
                    best_p = i
                p_list.append(i)
                dist_p_list.append(dist_1p)
                dist_inf_list.append(dist_pl)
                diff_list.append(abs(dist_1p - dist_pl))  
        print('Transition point: {:.2f}'.format(best_p))

        limit_output(
            p_list,
            dist_p_list,
            dist_inf_list,
            diff_list,
            "limits.csv"
        )

    # Compute equivalent P given a already existing consensus
    elif args.i:
        cons = np.genfromtxt(args.i)
        print_consensus(cons)
        best = np.inf
        incr = 0.01
        for i in np.arange(1 + incr, p, incr):
            A, b = FormalisationMatrix(P_list, J_list, w, i, args.v)
            x, r, u = Lp(A, b, i)
            dist = np.linalg.norm(cons - x, i)
            if (dist > best):
                print('Not improving anymore, stopping!')
                break
            else:
                print('p = {:.2f}'.format(i))
                print('Distance = {:.4f}'.format(dist))
                print('Current best distance = {:.4f}'.format(best))
                best = dist

    # Aggregate using principle values
    elif args.pv == True:
        print("DEBUG INFO: Aggregating on Agent Principle")
        p_list, _, cons_list, _, _, cons_1, cons_l=  aggregate_all_p(P_list=PP_list, 
                                                        J_list=PJ_list,
                                                        w=Pw)        
        ## Defining a cut point to drop all rows where there are P's that are higher than this
    
        # Cut point is the limit where p is within epsilon of p=\infty
        print("Cons list is: ", cons_list)
        cut_point = 10
        incr = 0.1
        j = 0
        for i in np.arange(1 + incr, 10, incr):
            cons = cons_list[j]
            dist_1p = np.linalg.norm(cons_1 - cons, i)
            dist_pl = np.linalg.norm(cons_l - cons, i)
            j += 1
            # Note: Hard Coded \epsilon value
            if (abs(dist_1p - dist_pl) < 0.005):
                cut_point = i
                print('Not improving anymore, stopping!')
                break

        cut_list = [cons_list[i] for i in range(len(cons_list)) if p_list[i] <= cut_point]
        print("DEBUG: cut_list length is: ", len(cut_list))
        con_vals = [0, 0]
        for j in range(2):
            con_vals[j] = sum(i[j+1] for i in cut_list) / len(cut_list)
        
        con_p = 1.0 
        best_dist = 999
        for j in range(len(cons_list)):
            dist = [abs(cons_list[j][1] - con_vals[0]), abs(cons_list[j][2] - con_vals[1])]
            dist = sum(dist)
            if dist < best_dist:
                best_dist = dist
                # to convert from ordinal list num to corresponding p
                con_p = (j/10)+1

        print("DEBUG INFO: Finding best P")
        ## Defining a cut point to drop all rows where there are P's that are higher than this
        # TODO: Include finding limit P rather than hard coding
        cut_point = 3.8
        cut_list = [cons_list[i] for i in range(len(cons_list)) if p_list[i] <= cut_point]
        print("DEBUG: cut_list length is: ", len(cut_list))
        con_vals = [0, 0]
        for j in range(2):
            con_vals[j] = sum(i[j+1] for i in cut_list) / len(cut_list)
        print("DEBUG: Con vals are: ",con_vals)
        
        con_p = 1.0 
        best_dist = 999
        for j in range(len(cons_list)):
            dist = [abs(cons_list[j][1] - con_vals[0]), abs(cons_list[j][2] - con_vals[1])]
            dist = sum(dist)
            if dist < best_dist:
                best_dist = dist
                # to convert from ordinal list num to corresponding p
                con_p = (j/10)+1

        print("DEBUG: Nearest P is: ", con_p)

        # TODO: you arent running aggregation with con_p
        print("DEBUG: Running Aggregation with P = ", con_p)
        agg_action_p_list, _, agg_action_cons_list, _, _, cons_actions = aggregate_principles(False, "aggregated_action_values.csv", con_p=con_p)
        agg_pref_p_list, _, agg_pref_cons_list, _, _, cons_prefs = aggregate_principles(True, "aggregated_preference_values.csv", con_p=con_p)

        decision = make_decision(cons_prefs, cons_actions)
        print("Decision made at ratio: ", decision[0], " to ", decision[1])

        # Check if explanation needed, if so, run
        #if args.ex != None:
            # Compute justification
    else:
        if p == 2:
            A, b = FormalisationMatrix(P_list, J_list, w, p, args.v)
            cons, r, u = L2(A, b)
            print_consensus(cons)
        elif p == 1:
            A, b = FormalisationMatrix(P_list, J_list, w, p, args.v)
            cons, r, u = L1(A, b)
            print_consensus(cons)
        elif p == -1:
            A, b = FormalisationMatrix(P_list, J_list, w, np.inf, args.v)
            cons, r, u = Linf(A, b)
            print_consensus(cons)
        else:
            A, b = FormalisationMatrix(P_list, J_list, w, p, args.v)
            cons, r, u = Lp(A, b, p)
            print_consensus(cons)
        if p != -1:
            print('U{} = {:.4f}'.format(p, u))
        else:
            print('U∞ = {:.4f}'.format(u))
        print()
        print('Residuals =', r)
        print('Max residual = {:.4f}'.format(np.max(r)))
        h, b = np.histogram(r, bins=np.arange(10))
        print('Residuals distribution =')
        print(np.vstack((h, b[:len(h)], np.roll(b, -1)[:len(h)])))
        if args.o:
            np.savetxt(args.o, cons, fmt='%.20f')
