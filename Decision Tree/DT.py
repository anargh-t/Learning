import pandas as pd
import math

df = pd.read_excel(r"C:\Users\anarg\Downloads\dtree dataset.xlsx")

df['BUYS_COMPUTER'] = df['BUYS_COMPUTER'].map({'no': 0, 'yes': 1})
df['AGE'] = df['AGE'].map({'<=30': 0, '31-40': 1, '>40': 2})
df['INCOME'] = df['INCOME'].map({'low': 0, 'medium': 1, 'high': 2})
df['STUDENT'] = df['STUDENT'].map({'no': 0, 'yes': 1})
df['CREDIT'] = df['CREDIT'].map({'fair': 0, 'excellent': 1})

# Function to calculate entropy
def calculate_entropy(df):
    total_count = len(df)
    if total_count == 0:
        return 0

    label_count = df['BUYS_COMPUTER'].value_counts()

    entropy = 0.0
    for count in label_count:
        probability = count / total_count
        entropy -= probability * math.log2(probability)

    return entropy

# Function to calculate information gain
def calculate_information_gain(df, feature):
    total_entropy = calculate_entropy(df)

    subsets = df.groupby(feature)

    weighted_entropy_sum = sum((len(subset) / len(df)) * calculate_entropy(subset) for _, subset in subsets)

    information_gain = total_entropy - weighted_entropy_sum
    return information_gain


# Finding the best feature to split on (root node)
def find_best_feature(df):
    best_gain = -float('inf')
    best_feature = None

    features = df.columns[:-1]  # Exclude label column

    for feature in features:
        gain = calculate_information_gain(df, feature)
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    return best_feature

# Function to build decision tree recursively
def build_decision_tree(df):
    # Base case: if all labels are the same, return that label
    if len(df['BUYS_COMPUTER'].unique()) == 1:
        return df['BUYS_COMPUTER'].iloc[0]

    # Calculate the best feature to split on
    best_feature = find_best_feature(df)

    # Create a dictionary to hold subtrees
    tree = {best_feature: {}}

    # Split the data based on the best feature's values and build subtrees recursively
    for value in df[best_feature].unique():
        subset = df[df[best_feature] == value]
        subtree = build_decision_tree(subset)
        tree[best_feature][value] = subtree

    return tree

# Calculate and print results for the first split
entropy_before_split = calculate_entropy(df)
best_feature_first_split = find_best_feature(df)

print("\n--- Summary ---")
print(f"Entropy before split: {entropy_before_split:.4f}")
print(f"Best feature to split on: {best_feature_first_split}")

# Build and print the decision tree
decision_tree = build_decision_tree(df)
print("\n--- Decision Tree ---")
print(decision_tree)
