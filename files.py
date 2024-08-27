import csv


def output_file(p, U, cons, dist_1, dist_l, v, name):
    """
    This function writes the results of solving the lp-regression for different p's.
    INPUT: p -- int, U -- list (Up distance function values), cons -- list of list
           (values of the consensus achieved per p), dist_1 -- list (distance from
           the consensus achieved for p=1 and the one for current p), dist_l -- list
           (distance from the consensus achieved for p=inf and the one for current p),
           v -- boolean (parameter, when v = True, we solve the prefference aggregation
           over moral values, when v = False, we solve the aggregation of moral values),
           name -- str (name of the file)
    """
    csv_rows = []
    if v:
        # header for the value prefference aggregation
        header = [
            "p", "Up", "Dist1", "Distl", "Rel-Rel",
            "Rel-Nonrel", "Nonrel-Rel", "Nonrel-Nonrel"
        ]
    else:
        # header for the aggregation of moral values
        header = [
            "p", "Up", "Dist1", "Distl", "Rel_adp_p",
            "Rel_div_p", "Nonrel_adp_p", "Nonrel_div_p",
            "Rel_adp_n", "Rel_div_n", "Nonrel_adp_n", "Nonrel_div_n"
        ]
    csv_rows.append(header)
    for i in range(len(p)):
        el = [p[i], U[i], dist_1[i], dist_l[i]]
        for j in range(len(cons[i])):
            # writing consensus
            el.append(cons[i][j])
        csv_rows.append(el)
    with open(name, 'w', newline='') as csvfile:
        # writing file
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    return None


def simple_output_file(p, y, name):
    """
    This function writes a measure for each p in a file.
    INPUT: p -- int, y -- list (measure), name -- str (name of the file)
    """
    header = ["p", "y(p)"]
    csv_rows = [header]
    for i in range(len(p)):
        el = [p[i], y[i]]
        csv_rows.append(el)
    with open(name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    return None

def limit_output(p, dist_p_list, dist_inf_list, diff_list, name):
    """
    This function writes the values computed finding the limit P, and transition points (-t True)
    INPUT: p -- int, y -- list (measure), name -- str (name of the file)
    """
    header = ["p", "Dist_p", "Dist_inf", "Diff"]
    csv_rows = [header]
    for i in range(len(p)):
        el = [p[i], dist_p_list[i], dist_inf_list[i], diff_list[i]]
        csv_rows.append(el)
    with open(name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
    return None