import argparse as ap
import numpy as np
import os
from matrices import FormalisationObjects, FormalisationMatrix
from files import output_file, limit_output
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

import juliapkg
juliapkg.require_julia("=1.10.3")
juliapkg.resolve()
from juliacall import Main as jl

np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=4)

def aggregate_all_p(P_list, J_list, w):
    """
    Used by pv argument to aggregate on principles in solve.py
    """
    A, b = FormalisationMatrix(P_list, J_list, w, 1, True)
    cons_1, _, ua = L1(A, b)
    print(cons_1)
    cons_1 = cons_1[1:3]
    print(cons_1)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, True)
    cons_l, _, _, = Linf(A, b)
    print(cons_l)
    cons_l = cons_l[1:3]
    print(cons_l)
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
        cons = cons[1:3]
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
    #prob.solve(solver='ECOS', verbose=False)
    prob.solve(solver='CPLEX', verbose=True)
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

def mLp(A, b, ps, λs, weight=True):
    wps = [λ / Lp(A, b, p) if weight else λ for λ, p in zip(λs, ps)]
    v = A.shape[1]
    x = cp.Variable(v)
    cost = cp.sum([wp * cp.pnorm(A @ x - b, p) for wp, p in zip(wps, ps)])
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver="ECOS", verbose=True)
    res = np.abs(A @ x.value - b)
    psi = np.var([wp * np.linalg.norm(res, p) for wp, p in zip(wps, ps)])
    return x.value, res, prob.value / sum(wps), psi


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-n', type=int, default=7, help='n')
    parser.add_argument('-m', type=int, default=2, help='m')
    parser.add_argument('-p', type=float, default=10, help='p')
    parser.add_argument('-e', type=float, default=1e-4, help='e')
    parser.add_argument(
        '-f',
        type=str,
        default="/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/22-01-2025-agent-data.csv",
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
    parser.add_argument('-o', type=str, help='write consensus to file', default="consensus_args_o.csv")
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
        #default='test-prefs.csv',
        default='none',
        help='store results in csv')
    
    parser.add_argument(
        '-pf',
        type=str,
        default=None,
        #default='/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/principle_test_cases/16-01-2025-ESS-principles.csv',
        help='CSV file with principle data'
    )    
    parser.add_argument(
        '-pv',
        type=bool,
        default=False,
        help='Compute the P value consensus aggregation method'
    )
    parser.add_argument(
        '-sml',
        type=bool,
        default=True,
        help="Generate consensus using the method described by Salas-Molina et al."
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
    if args.sml:
        print("In args.sml, computing Salas-Molina et al. method")
        file_path = '/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/means_and_salas_molina_ps/principles_for_slm.csv'
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"Warning: {file_path} is empty.")
        except pd.errors.EmptyDataError:
            print(f"Error: {file_path} is empty or not a valid CSV.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        print("DEBUG: df: ", df.head())
        consensus_df = pd.DataFrame()
        for series_name, series in df.items():
            print("DEBUG: series_name: ", series_name)
            print("DEBUG: series: ", series)
            p = series
            ps = np.atleast_1d(p)
            ps = np.where(ps == -1, np.inf, ps)
            λs = np.ones_like(ps)
            nλs = min(len(λs), len([]))
            λs[:nλs] = [][:nλs]

            P_list, J_list, w, country_dict = FormalisationObjects(
                filename=args.f, delimiter=',', weights=args.w)
            A, b = FormalisationMatrix(P_list, J_list, w, 1, args.v)
            # w has weights equal to 1, shape needs to be equal
            w = np.repeat(w, A.shape[1])

            cons, res, u, psi = mLp(A, b, ps, λs, False)

            P_list, J_list, w, country_dict = FormalisationObjects(
                filename=args.f, delimiter=',', weights=args.w)
            A_1, b_1 = FormalisationMatrix(P_list, J_list, w, 1, not(args.v))
            # w has weights equal to 1, shape needs to be equal
            w = np.repeat(w, A_1.shape[1])

            cons_1, res, u, psi = mLp(A_1, b_1, ps, λs, False)
            # mLp returns: x.value, res, prob.value / sum(wps), psi
            # NOTE: COLUMN NAMES ARE ONLY ACCURATE WHEN ARGS.V = FALSE
            consensus_df = consensus_df._append({
                'series_name': series_name,
                'Rel_div_p': cons[0],
                'Nonrel_div_p': cons[1],
                'Rel_div_n': cons[2],
                'Nonrel_div_n': cons[3],
                'Rel-Rel': cons_1[0],
                'Rel-Nonrel': cons_1[1],
                'Nonrel-Rel': cons_1[2],
                'Nonrel-Nonrel': cons_1[3]
            }, ignore_index=True)
            print("len of cons: ", len(cons))
            print("len of cons_1: ", len(cons_1))
        consensus_df.to_csv('/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/ess_example_data/single_example_results/single_example/means_and_salas_molina_ps/test_cons.csv', index=False)

    elif args.l:
        print("In args.l, computing Limit")
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
        print("In args.g, computing personal aggregation")
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
        print("In args.t, computing Threshold")

        # Join the two consensus lists of preferences and action judgements
        #   to get a single consensus list for p=1 and p=\infty
        A, b = FormalisationMatrix(P_list, J_list, w, 1, args.v)
        cons_1, r_1, u_1 = L1(A, b)
        cons_1 = cons_1[1:3]
        A, b = FormalisationMatrix(P_list, J_list, w, 1, not(args.v))
        cons_1_1, r_1_1, u_1_1 = L1(A, b)
        cons_1_1 = cons_1_1[1:3]
        cons_1 = np.concatenate((cons_1, cons_1_1))
        
        A, b = FormalisationMatrix(P_list, J_list, w, np.inf, args.v)
        cons_l, r_l, u_l = Linf(A, b)
        cons_l = cons_l[1:3]
        A, b = FormalisationMatrix(P_list, J_list, w, np.inf, not(args.v))
        cons_l_1, r_l_1, u_l_1 = Linf(A, b)
        cons_l_1 = cons_l_1[1:3]
        cons_l = np.concatenate((cons_l, cons_l_1))

        diff = np.inf
        incr = 0.1

        p_list = []
        dist_p_list = []
        dist_inf_list = []
        diff_list = []
        for i in np.arange(1 + incr, p, incr):
            A, b = FormalisationMatrix(P_list, J_list, w, i, args.v)
            cons, r, u = Lp(A, b, i)
            cons = cons[1:3]
            A, b = FormalisationMatrix(P_list, J_list, w, i, not(args.v))
            cons_cons, r_cons, u_cons = Lp(A, b, i)
            cons_cons = cons_cons[1:3]
            cons = np.concatenate((cons, cons_cons))
            print('p: {:.2f}, cons: '.format(i), cons)

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
        print("INFO: Aggregating on Agent Principle")
        p_list, _, cons_list, _, _, cons_1, cons_l=  aggregate_all_p(P_list=PP_list, 
                                                        J_list=PJ_list,
                                                        w=Pw)        
        ## Defining a cut point to drop all rows where there are P values whose distance is within epsilon of p=\infty
        cut_point = 10
        incr = 0.1
        j = 0
        epsilon = 0.005
        print(cons_list)
        for i in np.arange(1 + incr, 10, incr):
            cons = cons_list[j]
            dist_1p = np.linalg.norm(cons_1 - cons, i)
            dist_pl = np.linalg.norm(cons_l - cons, i)
            j += 1
            # Note: Hard Coded \epsilon value
            if (abs(dist_1p - dist_pl) < epsilon):
                cut_point = i
                print('Not improving anymore, stopping!')
                break
        print("DEBUG: Cut point is: ", cut_point)
        # Cut point defined, now cut from the list of consensus values, and find the mean of these
        cut_list = [cons_list[i] for i in range(len(cons_list)) if p_list[i] <= cut_point]
        con_vals = [sum(i[0] for i in cut_list) / len(cut_list), sum(i[1] for i in cut_list) / len(cut_list)]        
        con_p = 1.0 
        best_dist = 999
        for j in range(len(cut_list)):
            dist = [abs(cut_list[j][0] - con_vals[0]), abs(cut_list[j][1] - con_vals[1])]
            dist = sum(dist)
            if dist < best_dist:
                best_dist = dist
                # to convert from ordinal list num to corresponding p
                con_p = (j/10)+1
        print("DEBUG: Nearest P to mean con_vals is: ", con_p)

        ## Find the mean of all the P values
        principle_df = pd.read_csv(args.pf)
        principle_df = principle_df.drop(columns=['country'])
        mean_values = principle_df.mean(axis=0)
        print("Mean of each column in principle_df:")
        print(mean_values)
        egal = mean_values['rel'] / (mean_values['rel'] + mean_values['nonrel'])
        principle = 1 + 9 * egal
        print("Principle value via means is: ", principle)
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
