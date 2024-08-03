import argparse as ap
import numpy as np
import os
from matrices import FormalisationObjects, FormalisationMatrix
from files import output_file, limit_output
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math

np.set_printoptions(edgeitems=1000, linewidth=1000, suppress=True, precision=4)

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

    cons_prefs: array that maps to [p, ub, cons, dist_1p, dist_pl]
    cons_actions: array that maps to [p, ub, cons, dist_1p, dist_pl]
    cons in this case is in format [rel-rel, rel-nonrel, nonrel-rel,nonrel-nonrel], and similar for actions
    """
    adp = (cons_prefs[2][1] * cons_actions[2][1]) + (cons_prefs[2][2] * cons_list[2][2])
    div = 1 - adp
    decision = [adp, div]
    return decision

# TODO: Unfinished
def compute_justification(agent_ID, personal_vals, principle_vals, cons_row, decision) -> str:
    """
    This function takes in an agents ID and consensus personal+principle value systems and returns a text understanding of their justification
    This function takes in the consensus personal+principle value systems and constructs a string justification 

    agent_ID: the ID of the agent, leave as None if group justification wanted
    personal_vals: original personal values file
    principle_vals: original principle values file
    cons_row: the row in the aggregation that matches the consensus
    decision: list [adp, div] that represents the decision made as a ratio of 2 scores
    """
    justification = []
    principle_explainer = {
        "util_justification": "which is justified as the principle most representative of the group attempts to maximise the total happiness of the group",
        "egal_justification": "which is justified as the principle most representative of the group attempts to maximise equality"
    }

    personal_data = pd.read_csv(personal_vals)
    principle_data = pd.read_csv(principle_vals)

    print("DEBUG: priciple vals are: " + principle_vals)
    print("DEBUG: personal_vals are: " + personal_vals)
    
    # Get consensus items (used for both)
    cons_principle_val = cons_row[1] * 100
    if cons_principle_val < 50:
        cons_principle = "utilitarianism"
        cons_principle_val = 100 - cons_principle_val  
    else: 
        cons_principle = "egalitarianism"
    
    if agent_ID != None:
        # Agent specific Justification
        # Get whether the agent prefered util/egal
        principle_percentage = principle_vals.iloc[agent_ID]["rel"]*100
        if principle_percentage < 50:
            principle = "utilitarianism"
            principle_percentage = 100 - principle_percentage  
        else: 
            principle = "egalitarianism"
        justification.append("You Prefer ", principle, " ", principle_percentage, "% of the time.")

        justification.append("The consenus princple prefers ", cons_principle, " ", cons_principle_val,"%.")

        justification.append("The decision has been made to ")


    else:
        # General (outsider) justification
        justification.append("The group prefers ")


    return justification

def fill_prinicples(personal_vals, principle_vals) -> pd.DataFrame:
    """
    This function takes in the principle and personal value data, and performs clustering on the personal data.
    It then fills in any missing principle values with the mean of the cluster.  
    """
    personal_data = pd.read_csv(personal_vals)
    principle_data = pd.read_csv(principle_vals)

    # Check if any principles need to be filled, otherwise return
    if not principle_data['rel'].isna().any():
        return principle_data
    # Perform clustering on personal data
    X = personal_data.drop('country', axis=1).values
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(scaler.fit_transform(X))
    # TODO: use the elbow method to pick number of clusters, default=3
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_scaled)
    labels = kmeans.labels_
    print(labels)

    cluster_averages = []
    for i in range(kmeans.n_clusters):
        cluster_countries = personal_data['country'][labels == i]
        principle_values = principle_data[principle_data['country'].isin(cluster_countries)]
        # Drop all values that don't have principles
        principle_values = principle_values.dropna()
        cluster_average = principle_values.drop('country', axis=1).values.mean(axis=0)
        cluster_averages.append(cluster_average)

    for index, row in principle_data.iterrows():
        if math.isnan(row['rel']):
            # Get the personal value system for the country
            personal_value = personal_data.loc[personal_data['country'] == row['country']]
            scaled_personal_value = scaler.transform(personal_value.drop('country', axis=1).values)
            cluster = kmeans.predict(scaled_personal_value)[0]  # [0] to get the single value
            keys = ['rel', 'nonrel', 'a_adp_rel', 'a_div_rel', 'a_adp_nonrel', 'a_div_nonrel']
            for i, key in enumerate(keys):
                value = cluster_averages[cluster][i]
                principle_data.at[index, key] = value

    return principle_data

def aggregate_values(aggregation_type, filename, con_p=0.0, 
                     P_list=None, 
                     J_list=None, 
                     w=None,
                     principle_val=1
                     ):
    """
    We run aggregation of the action value matrices and store in a file
    """
    
    consensus_vals = []
    print("DEBUG: CON_P IS ", con_p)

    ## The same as in elif args.g
    A, b = FormalisationMatrix(P_list, J_list, w, 1, aggregation_type)
    cons_1, _, ua = L1(A, b)
    A, b = FormalisationMatrix(P_list, J_list, w, np.inf, aggregation_type)
    cons_l, _, _, = Linf(A, b)
    dist_1p = np.linalg.norm(cons_1 - cons_1, 1)
    dist_pl = np.linalg.norm(cons_l - cons_1, np.inf)
    
    p = principle_val
    
    print('{:.2f} \t \t {:.4f}'.format(p, ua))
    incr = 0.1
    p_list = [1.0]
    u_list = [ua]
    cons_list = [cons_1]
    dist_1p_list = [dist_1p]
    dist_pl_list = [dist_pl]

    # ATM this just returns you the P val consensus, not the action consensus
    """
    while p < principle_val:
        p += incr
        A, b = FormalisationMatrix(P_list, J_list, w, p, aggregation_type)
        cons, _, ub = Lp(A, b, p)
        p_list.append(p)
        u_list.append(ub)
        cons_list.append(cons)
        dist_1p = np.linalg.norm(cons_1 - cons, p)
        dist_pl = np.linalg.norm(cons_l - cons, p)
        dist_1p_list.append(dist_1p)
        dist_pl_list.append(dist_pl)
        print('{:.2f} \t \t {:.4f}'.format(p, ub))
        if math.isclose(p, float(con_p), rel_tol=1e-9):
            print("DEBUG: P and con_P equal")
            consensus_vals = [p, ub, cons, dist_1p, dist_pl]
    """
    A, b = FormalisationMatrix(P_list, J_list, w, p, aggregation_type)
    cons, _, ub = Lp(A, b, p)
    p_list.append(p)
    u_list.append(ub)
    cons_list.append(cons)
    dist_1p = np.linalg.norm(cons_1 - cons, p)
    dist_pl = np.linalg.norm(cons_l - cons, p)
    dist_1p_list.append(dist_1p)
    dist_pl_list.append(dist_pl)
    print('{:.2f} \t \t {:.4f}'.format(p, ub))
    if math.isclose(p, float(con_p), rel_tol=1e-9):
        print("DEBUG: P and con_P equal")
        consensus_vals = [p, ub, cons, dist_1p, dist_pl]
    print("DEBUG: Saving to file, ", filename)
    output_file(
        p_list,
        u_list,
        cons_list,
        dist_1p_list,
        dist_pl_list,
        aggregation_type,
        filename)
    
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
    print("obj value:", obj)
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
    print("obj value: ", obj)
    r = np.abs(A @ cons - b)
    return cons, r, np.linalg.norm(r, np.inf)


def IRLS(A, b, p, max_iter=int(1e6), e=1e-3, d=1e-4):
    """
    This function runs the IRLS method for finding consensus for any P >= 3
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
    if p >= 2:  # pIRLS implementation (NIPS 2019)
        from julia import Main
        # from julia import IRLSmod
        Main.include(
            os.path.dirname(
                os.path.realpath(__file__)) +
            '/IRLS-pNorm.jl')
        # constraints needed for pIRLS (empty)
        C = np.zeros_like(A)
        d = np.zeros_like(b)
        epsilon = 1e-10
        cons, it = Main.pNorm(epsilon, A, b.reshape(-1, 1),
                              p, C, d.reshape(-1, 1))
        # cons, it = IRLS.pNorm(epsilon, A, b.reshape(-1, 1), p, C, d.reshape(-1, 1))
        r = np.abs(A @ cons - b)
        return cons, r, np.linalg.norm(r, p)
    else:  # vanilla IRLS implementation
        return IRLS(A, b, p)


### TODO: Make a new arguement that allows user to compute preference aggregation using principles
if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-n', type=int, default=7, help='n')
    parser.add_argument('-m', type=int, default=2, help='m')
    parser.add_argument('-p', type=float, default=10, help='p')
    parser.add_argument('-e', type=float, default=1e-4, help='e')
    parser.add_argument(
        '-f',
        type=str,
        default='/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/toy_data.csv',
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
    parser.add_argument('-o', type=str, help='write consensus to file', default="consensus.csv")
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
        default='none',
        #default='results.csv',
        help='store results in csv')
    
    parser.add_argument(
        '-pf',
        type=str,
        default='/home/ia23938/Documents/GitHub/ValueSystemsAggregation/data/toy_principles.csv',
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
        default="A",
        help="Generate explanation of agent with index as arg"
    )

    args = parser.parse_args()

    p = args.p
    n = args.n
    m = args.m

    # init for personal aggregation

    P_list, J_list, w, country_dict = FormalisationObjects(
        filename=args.f, delimiter=',', weights=args.w)
    
    # init for principle aggregation
    if args.pv == True and args.pf != 'none':
        # Solve missing principle values
        fill_prinicples(personal_vals=args.f, principle_vals=args.pf)
        PP_list, PJ_list, Pw, Pcountry_dict = FormalisationObjects(
            filename=args.pf, delimiter=',', weights=args.w)

    # Compute the limit P
    if args.l:
        A, b = FormalisationMatrix(P_list, J_list, w, 1, args.v)
        cons_1, _, _, = L1(A, b)
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
        cons_1, _, ua = L1(A, b)
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
            print('{:.2f} \t \t {:.4f}'.format(p, ub))

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
        A, b = FormalisationMatrix(P_list, J_list, w, 1, args.v)
        cons_1, r_1, u_1 = L1(A, b)
        A, b = FormalisationMatrix(P_list, J_list, w, np.inf, args.v)
        cons_l, r_l, u_l = Linf(A, b)
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
                break
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
        p_list, _,  cons_list, _, _, _= aggregate_values(True, "consensus_principles.csv")
        
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
        agg_action_p_list, _, agg_action_cons_list, _, _, cons_actions = aggregate_values(False, "aggregated_action_values.csv", con_p=con_p)
        agg_pref_p_list, _, agg_pref_cons_list, _, _, cons_prefs = aggregate_values(True, "aggregated_preference_values.csv", con_p=con_p)

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
