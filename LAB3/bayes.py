from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

bayesNet = BayesianModel([('L', 'N'), ('S', 'N'), ('I', 'N'), ('S', 'R'), ('N', 'R')])

cpd_l = TabularCPD(variable='L', variable_card=2, values=[[0.92], [0.08]])
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.79], [0.21]])
cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.88], [0.12]])
cpd_n = TabularCPD(variable='N', variable_card=2, 
                   values=[[0.97, 0.83, 0.92, 0.78, 0.73, 0.21, 0.12, 0.03], 
                           [0.03, 0.17, 0.08, 0.22, 0.27, 0.79, 0.88, 0.97]],
                   evidence=['L', 'S', 'I'],
                   evidence_card=[2, 2, 2])
cpd_r = TabularCPD(variable='R', variable_card=2, 
                   values=[[0.95, 0.92, 0.92, 0.62], 
                           [0.05, 0.08, 0.08, 0.38]],
                   evidence=['N', 'S'],
                   evidence_card=[2, 2])

bayesNet.add_cpds(cpd_l, cpd_i, cpd_s, cpd_n, cpd_r)

print("Model check result: ", bayesNet.check_model())

solver = VariableElimination(bayesNet)

# Querying the network for probabilities

prob_N = solver.query(variables=['N'])
print("Probability of N (not scored):", prob_N)

# Probability of 'N' given 'L' is True (ML model signaled)
prob_N_given_L = solver.query(variables=['N'], evidence={'L': 1})
print("Probability of N (not scored) given L (ML signaled):", prob_N_given_L)

# Finding independencies
independencies = bayesNet.get_independencies()
print("Independencies in the network:", independencies)
