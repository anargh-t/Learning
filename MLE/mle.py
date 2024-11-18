
import math

# df = pd.read_excel(r"C:\Users\anarg\Desktop\MLE.xlsx")
# print(df)


infected = [1, 0, 1, 1, 0, 1, 1, 0, 0, 1]


infected_count = sum(infected)
total_observations = len(infected)
not_infected_count = total_observations - infected_count

# Maximum Likelihood Estimate for p
p_mle = infected_count / total_observations


# Log-Likelihood Function
log_likelihood = (
    infected_count * math.log(p_mle) +
    not_infected_count * math.log(1 - p_mle)
)

print(f"Number of infected individuals: {infected_count}")
print(f"Total number of observations: {total_observations}")
print(f"Estimated value of p (MLE): {p_mle}")
print(f"Maximum Log-Likelihood: {log_likelihood}")