import numpy as np
import pandas as pd

def PMatrix(df_row):
    rel = df_row['rel']/(df_row['rel']+df_row['nonrel'])
    per = df_row['nonrel']/(df_row['rel']+df_row['nonrel'])
    P = [[0,rel],[per,0]]
    return np.array(P)

def JMatrixs(df_row):
    J_p = [[df_row['a_adp_rel'],df_row['a_div_rel']],[df_row['a_adp_nonrel'],df_row['a_div_nonrel']]]
    J_n = [[-df_row['a_adp_rel'],-df_row['a_div_rel']],[-df_row['a_adp_nonrel'],-df_row['a_div_nonrel']]]
    return np.array(J_p),np.array(J_n)

def Weights(df,n_countries,weights = 0):
    w = []
    n_total = 0
    if weights == 1:
        for i in range(n_countries):
            n_participants = df.iloc[i]['rel'] + df.iloc[i]['nonrel']
            n_total += n_participants
            w.append(n_participants)
        w = np.array(w)
        w_norm = []
        for i in range(n_countries):
            #print(w[i])
            w_norm.append(w[i] / n_total)
            #print(w_norm[i])
        return np.array(w_norm)
    elif weights == 2:
        for i in range(n_countries):
            n_participants = df.iloc[i]['population']
            n_total += n_participants
            w.append(n_participants)
        w = np.array(w)
        w_norm = []
        for i in range(n_countries):
            #print(w[i])
            w_norm.append(w[i] / n_total)
            #print(w_norm[i])
        return np.array(w_norm)
    else:
        for i in range(n_countries):
            w.append(1)
        return np.array(w)


def FormalisationObjects(filename = 'AAMAS2022.csv',delimiter=',',weights = 0):

    df = pd.read_csv(filename, delimiter=delimiter)
    n_countries = df.shape[0] # number of rows

    J_list = []
    P_list = []
    country_dict = {}
    for i in range(n_countries):
        country = df.iloc[i]['country']
        country_dict.update({i:country})
        P = PMatrix(df.iloc[i])
        J_p,J_n = JMatrixs(df.iloc[i])
        P_list.append(P)
        J_list.append((J_p,J_n))

    w = Weights(df,n_countries,weights)
    return P_list,J_list,w,country_dict


def IMatrix(dim = 2):
    I = []
    for i in range(dim):
        I_row = []
        for j in range(dim):
            if i==j:
                I_row.append(1)
            else:
                I_row.append(0)
        I.append(I_row)
    return np.array(I)

def BMatrix(w, n_val = 2, n_actions = 2 ):
    I = IMatrix(dim = 2*n_val*n_actions)
    B = np.array(w[0]*I)
    for i in range(1,len(w)):
        B = np.concatenate((B,w[i]*I))
    return B

def Vectorisation(M):
    vector = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            vector.append(M[i][j])
    return vector

def BVector(J_list, w):
    b = []
    for i in range(len(w)):
        j_p = Vectorisation(J_list[i][0])
        j_n = Vectorisation(J_list[i][1])
        for k in range(len(j_p)):
            b.append(w[i]*j_p[k])
        for k in range(len(j_n)):
            b.append(w[i]*j_n[k])
        
    return np.array(b)

def CMatrix(w, n_val = 2):
    I = IMatrix(dim = n_val*n_val)
    C = np.array(w[0]*I)
    for i in range(1,len(w)):
        C = np.concatenate((C,w[i]*I))
    return C

def CVector(P_list, w):
    c = []
    for i in range(len(w)):
        p = Vectorisation(P_list[i])
        for k in range(len(p)):
            c.append(w[i]*p[k])

    return np.array(c)
        
def FormalisationMatrix(P_list,J_list,w):
    B = BMatrix(w, n_val = J_list[0][0].shape[0], n_actions = J_list[0][0].shape[1] )
    b = BVector(J_list, w)
    C = CMatrix(w, n_val = P_list[0][0].shape[0])
    c = CVector(P_list, w)

    return B,b,C,c

def BMatrix2(w, n_val = 2, n_actions = 2, p = 2):
    I = IMatrix(dim = 2*n_val*n_actions)
    B = np.array((w[0]**(1/p))*I)
    for i in range(1,len(w)):
        B = np.concatenate((B,(w[i]**(1/p))*I))
    return B


def BVector2(J_list, w, p = 2):
    b = []
    for i in range(len(w)):
        j_p = Vectorisation(J_list[i][0])
        j_n = Vectorisation(J_list[i][1])
        for k in range(len(j_p)):
            b.append((w[i]**(1/p))*j_p[k])
        for k in range(len(j_n)):
            b.append((w[i]**(1/p))*j_n[k])
        
    return np.array(b)

def CMatrix2(w, n_val = 2, p = 2):
    I = IMatrix(dim = n_val*n_val)
    C = np.array((w[0]**(1/p))*I)
    for i in range(1,len(w)):
        C = np.concatenate((C,(w[i]**(1/p))*I))
    return C

def CVector2(P_list, w, p = 2):
    c = []
    for i in range(len(w)):
        pref = Vectorisation(P_list[i])
        for k in range(len(pref)):
            c.append((w[i]**(1/p))*pref[k])

    return np.array(c)
        
def FormalisationMatrix2(P_list,J_list,w,p = 2,v=True):
    if v:
        A = CMatrix2(w, n_val = P_list[0][0].shape[0],p=p)
        b = CVector2(P_list, w,p=p)
    else:
        A = BMatrix2(w, n_val = J_list[0][0].shape[0], n_actions = J_list[0][0].shape[1],p=p)
        b = BVector2(J_list, w,p=p)
        
    return A,b





if __name__ == '__main__':
    P_list,J_list,w,country_dict = FormalisationObjects()
    B,b,C,c = FormalisationMatrix(P_list,J_list,w)

    print(c)