import pandas as pd

file_path = r"C:\Users\anarg\Downloads\spam_emails.xlsx"
df = pd.read_excel(file_path)

df['text'] = df['text'].fillna('')

# Preprocess text
df['text'] = df['text'].str.replace(r"[^A-Za-z ]", "", regex=True)
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.split()

# Convert labels to numeric values
df['label'] = df['label'].replace(['not spam', 'spam'], [0, 1]).astype(int)

messages = df['text']
labels = df['label']

spam_messages = messages[labels == 1]
ham_messages = messages[labels == 0]

spam_word_count = {}
ham_word_count = {}

for message in spam_messages:
    if isinstance(message, list):
        for word in message:
            spam_word_count[word] = spam_word_count.get(word, 0) + 1

for message in ham_messages:
    if isinstance(message, list): 
        for word in message:
            ham_word_count[word] = ham_word_count.get(word, 0) + 1

# Calculate priors for spam and ham
priori_p_spam = len(spam_messages) / len(messages)
priori_p_ham = len(ham_messages) / len(messages)

# Calculate total words in spam and ham
total_spam_words = sum(spam_word_count.values())
total_ham_words = sum(ham_word_count.values())

def calculate_word_likelihood(word, word_count_dict, total_words, smoothing_factor=1):
    return (word_count_dict.get(word, 0) + smoothing_factor) / (total_words + smoothing_factor * len(word_count_dict))

def classify_email(mail, spam_word_count, ham_word_count, total_spam_words, total_ham_words, priori_p_spam, priori_p_ham):
    mail_words = mail.lower().split()

    likelihood_spam = 1.0
    likelihood_ham = 1.0

    # Calculate likelihood for each word in the mail
    for word in mail_words:
        likelihood_spam *= calculate_word_likelihood(word, spam_word_count, total_spam_words)
        likelihood_ham *= calculate_word_likelihood(word, ham_word_count, total_ham_words)

    # Calculate posterior probabilities
    posterior_spam = likelihood_spam * priori_p_spam
    posterior_ham = likelihood_ham * priori_p_ham

    # Normalize the probabilities to prevent zero division
    total_prob = posterior_spam + posterior_ham
    if total_prob == 0:
        print("Unable to classify the email (both probabilities are zero).")
        return

    # Normalize to get final posterior probabilities
    posterior_spam /= total_prob
    posterior_ham /= total_prob

    if posterior_spam > posterior_ham:
        print("The email is classified as spam.")
    else:
        print("The email is classified as not spam.")

print()

#Prediction
mail = input("Enter the email content: ")
classify_email(mail, spam_word_count, ham_word_count, total_spam_words, total_ham_words, priori_p_spam, priori_p_ham)
