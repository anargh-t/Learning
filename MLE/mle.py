import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\anarg\Desktop\MLE.xlsx")
print("Data Set: \n",df)

infected_count = df['Infected'].sum()
total_observations = len(df['Infected'])
not_infected_count = total_observations - infected_count

p_mle = infected_count / total_observations

log_likelihood = infected_count * np.log(p_mle) + not_infected_count * np.log(1 - p_mle)

print("Loglikelihood: ",log_likelihood)

print("Parameter that maximize the likelihood: ",p_mle)