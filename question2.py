import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

df = pd.read_csv('latestdata.csv', low_memory=False)
df['visiting_Wuhan'] = df['travel_history_location'].apply(lambda x: 1 if 'Wuhan' in str(x) else 0)
df['symptom_onset'] = df['date_onset_symptoms'].apply(lambda x: 0 if pd.isnull(x) else 1)
df['death'] = df['outcome'].apply(lambda x: 1 if x == 'died' else 0)
df['true_patient'] = df['symptom_onset'] 

# Define the structure of the Bayesian Network
model = BayesianNetwork([('visiting_Wuhan', 'symptom_onset'),
                         ('symptom_onset', 'true_patient'),
                         ('visiting_Wuhan', 'death')])

# Define the CPDs (Conditional Probability Distributions)
cpd_visiting_Wuhan = TabularCPD(variable='visiting_Wuhan', variable_card=2, values=[[0.99], [0.01]])
cpd_symptom_onset = TabularCPD(variable='symptom_onset', variable_card=2,
                               values=[[0.99, 0.3], [0.01, 0.7]],
                               evidence=['visiting_Wuhan'], evidence_card=[2])
cpd_true_patient = TabularCPD(variable='true_patient', variable_card=2,
                               values=[[0.999, 0.1], [0.001, 0.9]],
                               evidence=['symptom_onset'], evidence_card=[2])
cpd_death = TabularCPD(variable='death', variable_card=2,
                       values=[[0.999, 0.95], [0.001, 0.05]],
                       evidence=['visiting_Wuhan'], evidence_card=[2])

model.add_cpds(cpd_visiting_Wuhan, cpd_symptom_onset, cpd_true_patient, cpd_death)
model.check_model()
inference = VariableElimination(model)

# Q1: Probability of symptom onset if visited Wuhan
prob_symptom_onset_given_wuhan = inference.query(variables=['symptom_onset'], evidence={'visiting_Wuhan': 1})
print(prob_symptom_onset_given_wuhan)

# Q2: Probability of true patient if symptom onset and visited Wuhan
prob_true_patient_given_symptom_wuhan = inference.query(variables=['true_patient'], evidence={'symptom_onset': 1, 'visiting_Wuhan': 1})
print(prob_true_patient_given_symptom_wuhan)

# Q3: Probability of death if visited Wuhan
prob_death_given_wuhan = inference.query(variables=['death'], evidence={'visiting_Wuhan': 1})
print(prob_death_given_wuhan)

# Q4: Estimate the average recovery interval (using available data)
# Ensure consistent date format for parsing
df['date_death_or_discharge'] = pd.to_datetime(df['date_death_or_discharge'], errors='coerce', dayfirst=True)
df['date_onset_symptoms'] = pd.to_datetime(df['date_onset_symptoms'], errors='coerce', dayfirst=True)

# Calculate recovery times
recovery_times = (df['date_death_or_discharge'] - df['date_onset_symptoms']).dt.days
average_recovery_time = recovery_times.mean()
print(f'Average Recovery Time: {average_recovery_time} days')
