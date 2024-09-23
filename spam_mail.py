import pandas as pd

file_path = r"C:\Users\anarg\Downloads\spam_emails.xlsx"
df = pd.read_excel(file_path)

df['text'] = df['text'].str.replace(r"[^A-Za-z ]", "", regex=True)

df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.split()

df['label'] = df['label'].replace(['not spam', 'spam'], [0, 1])
# print(df['label'])
# print(df)

messages = df['text']
labels = df['label']

spam_messages = messages[labels==0]
ham_messages = messages[labels==1]

# print(spam_messages)
# print(ham_messages)

spam_word_count = {}
ham_word_count = {}

for i in spam_messages:
    words = str(i).split()
    for j in words:
        if j in spam_word_count:
            spam_word_count[j]+=1
        else:
            spam_word_count[j]=1
# print(spam_word_count)

for i in ham_messages:
    words = str(i).split()
    for j in words:
        if j in ham_word_count:
            ham_word_count[j]+=1
        else:
            ham_word_count[j]=1
# print(ham_word_count)

print("Total spam messages: ",len(spam_messages))
print("Total ham messages: ",len(ham_messages))

priori_p_spam = len(spam_messages) / len(messages)
print("priori probability of spam: ",priori_p_spam)
priori_p_ham = len(ham_messages)/len(messages)
print("priori probability of ham: ",priori_p_ham)

total_spam_words = sum(spam_word_count.values())
print("total spam words : ",total_spam_words)
total_ham_words = sum(ham_word_count.values())
print("total ham words: ",total_ham_words)



def calculate_word_likelihood(word, word_count_dict, total_words, smoothing_factor=1):
    # Add smoothing to avoid zero probabilities for unseen words (Laplace smoothing)
    return (word_count_dict.get(word, 0) + smoothing_factor) / (total_words + smoothing_factor * len(word_count_dict))

def classify_email(mail, spam_word_count, ham_word_count, total_spam_words, total_ham_words, priori_p_spam, priori_p_ham):
    mail_words = mail.lower().split()

    # Initialize likelihood probabilities
    likelihood_spam = 1.0
    likelihood_ham = 1.0

    for word in mail_words:
        likelihood_spam *= calculate_word_likelihood(word, spam_word_count, total_spam_words)
        likelihood_ham *= calculate_word_likelihood(word, ham_word_count, total_ham_words)

    posterior_spam = likelihood_spam * priori_p_spam
    posterior_ham = likelihood_ham * priori_p_ham

    # Normalize the probabilities (optional but gives more interpretable results)
    total_prob = posterior_spam + posterior_ham
    posterior_spam /= total_prob
    posterior_ham /= total_prob

    # Print probabilities and classify
    print(f"Posterior probability of spam: {posterior_spam:.4f}")
    print(f"Posterior probability of ham: {posterior_ham:.4f}")

    if posterior_spam > posterior_ham:
        print("The email is classified as spam.")
    else:
        print("The email is classified as not spam (ham).")

# Example usage
mail = input("Enter the email content: ")
classify_email(mail, spam_word_count, ham_word_count, total_spam_words, total_ham_words, priori_p_spam, priori_p_ham)


