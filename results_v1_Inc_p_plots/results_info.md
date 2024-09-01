# Format of results data
There are 25 groups
- each group has 4 different p values it must test per context
- each group goes through of 3 different contexts:
- each row is: context, p value, agent name, satisfaction score

Group 1 run their iteration. This runs from row 1 to row 48.
- row 1-16 is context 1, with 4 different p values
- this continues for each context up to a p value of 3


there are 10 rounds, so the 25 groups go through this process 10 times

0-74 is one experiment


row 0-11 is one single round
12 - 23 is second round

DEBUG: Experiment scores

e.g. format:
[
    [
        p=1,
        p=10,
        p=t,
        p=p
    ]
]

%%%%%%%%%%%%%%%%%%%%%
 [
    [
        {0: 0.314616903218605, 1: 0.39601946518923836, 2: 0.16897976996360825, 3: 0.48294020835304285}, 
        {0: 0.38763297213460113, 1: 0.46903553410523446, 2: 0.24199583887960435, 3: 0.555956277269039}, 
        {0: 0.3586254834328405, 1: 0.44002804540347384, 2: 0.21298835017784368, 3: 0.5269487885672782}, 
        {0: 0.3658624424916843, 1: 0.44726500446231765, 2: 0.22022530923668748, 3: 0.534185747626122}
    ],
 
    [
        {0: 0.14256180438787647, 1: 0.024224851820145277, 2: 0.02808046748062113, 3: 0.05899447006368819}, 
        {0: 0.13267076637656794, 1: 0.014333813808836735, 2: 0.01818942946931259, 3: 0.049103432052379645}, 
        {0: 0.14503612681363381, 1: 0.026699174245902624, 2: 0.030554789906378478, 3: 0.061468792489445534}, 
        {0: 0.13882150319052866, 1: 0.020484550622797457, 2: 0.02434016628327331, 3: 0.05525416886634037}
    ],
    
    [
        {0: 0.17301745432823143, 1: 0.5001398713294292, 2: 0.09715878009117923, 3: 0.09709265753818642}, 
        {0: 0.12460187536152029, 1: 0.5485554502961403, 2: 0.0487432011244681, 3: 0.048677078571475296}, 
        {0: 0.13363970796956703, 1: 0.5395176176880936, 2: 0.05778103373251485, 3: 0.057714911179522044}, 
        {0: 0.15059674476185422, 1: 0.5225605808958064, 2: 0.07473807052480202, 3: 0.07467194797180922}
    ]

]
%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%
DECISIONS:
debug:  [
    [
        [0.22034259271698586, 0.013006399354801204], 
        [0.06305957410307769, 0.057101927376922004], 
        [0.08246102366820196, 0.04858601192510803], 
        [0.09248628445405602, 0.04698879393427133]
    ], 
    [
        [0.0020342659393683057, -0.0601931901605178], 
        [0.09613671492466577, -0.12410478734346797], [0.004456010549852875, -0.09254668742297928], [-0.03476647214201755, -0.061344738385434396]
    ], 
    [
        [-0.1399367766669715, -0.08205251581271009], [-0.26368212503235827, -0.06702436336040477], [-0.20326414087956976, -0.07287940570011961], [-0.12187335409082808, -0.08631011847112835]
    ]
]