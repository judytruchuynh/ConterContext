import pandas as pd
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

def plot_form_responses_1(df, column_name):
    # Calculate the number and percentage of empty or 'No' responses
    empty_responses = len(df[df[column_name].isnull() ]) #| (df[column_name] == 'No')
    total_responses = len(df)
    count_no_response = empty_responses
    count_responded = total_responses - empty_responses

    # Create labels for the pie chart and bar chart
    labels = ['No Response', 'Responded']
    sizes = [count_no_response, count_responded]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot the pie chart
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title('Percentage of Form Responses')

    # Plot the bar chart
    ax2.bar(labels, sizes)
    ax2.set_ylabel('Count')
    ax2.set_title('Number of Form Responses')

    # Add annotations to the bars
    for bar in ax2.patches:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

    # Adjust spacing between subplots
    fig.subplots_adjust(wspace=0.9)

    # Display the charts
    plt.show()

    
    
def plot_comparison(df, new_df_column, df_column):
    # Count observations in new_df with the condition
    new_df_count = df[new_df_column].str[:5].nunique()

    # Count observations in df without applying the condition
    df_count = len(df)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Pie chart
    labels = [new_df_column, df_column]
    sizes = [new_df_count, df_count]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_aspect('equal')
    ax1.set_title('Observations Count Comparison')

    # Bar chart
    x = [new_df_column, df_column]
    y = [new_df_count, df_count]
    bars = ax2.bar(x, y, width=0.5)  # Adjust the bar width as desired
    ax2.set_ylabel('Comments Count')
    ax2.set_title('Observations Count Comparison')

    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

    # Increase the y-axis limit
    ax2.set_ylim(top=max(y) * 1.1)

    # Adjust the layout
    plt.tight_layout()

    # Display the combined chart
    plt.show()    
    

import re
def split_sentences(df, column_name):
    # Create a regex pattern to match sentences
    pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"

    # Split the values with multiple sentences into different observations
    split_values = df[column_name].str.split(pattern)

    # Create a new dataframe to store the split observations
    new_rows = []
    for i, value in enumerate(split_values):
        if len(value) > 1:
            current_code = f"FF{i+1:03d}"
            for j, sentence in enumerate(value):
                new_code = f"{current_code}{j+1:02d}"
                new_rows.append({
                    'Comment_ID': new_code,
                    column_name: sentence
                })
        else:
            current_code = f"FF{i+1:03d}"
            new_code = f"{current_code}00"
            new_rows.append({
                'Comment_ID': new_code,
                column_name: value[0]
            })

    new_df = pd.DataFrame(new_rows)

    return new_df



def plot_comparison(df, new_df_column, df_column):
    # Count observations in new_df with the condition
    new_df_count = df[new_df_column].str[:5].nunique()

    # Count observations in df without applying the condition
    df_count = len(df)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Pie chart
    labels = [new_df_column, df_column]
    sizes = [new_df_count, df_count]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_aspect('equal')
    ax1.set_title('Observations Count Comparison')

    # Bar chart
    x = [new_df_column, df_column]
    y = [new_df_count, df_count]
    bars = ax2.bar(x, y, width=0.5)  # Adjust the bar width as desired
    ax2.set_ylabel('Comments Count')
    ax2.set_title('Observations Count Comparison')

    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

    # Increase the y-axis limit
    ax2.set_ylim(top=max(y) * 1.1)

    # Adjust the layout
    plt.tight_layout()

    # Display the combined chart
    plt.show()




import matplotlib.pyplot as plt

def plot_observation_count(new_df_library, df_1, column_name):
    # Count observations in new_df_library with the condition
    new_df_count = new_df_library["Comment_ID"].str[:5].nunique()
    # Count observations in df_1 without applying the condition
    df_1_count = len(df_1)
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Pie chart
    labels = [column_name, 'other participants']
    sizes = [new_df_count, df_1_count]
    ax1.pie(sizes, autopct='%1.1f%%', startangle=90)
    ax1.set_aspect('equal')
    ax1.set_title('Participants Count Comparison')
    # Add a legend to the pie chart
    ax1.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=1)

    # Bar chart
    x = [column_name, 'total participants']
    y = [new_df_count, df_1_count]
    bars = ax2.bar(x, y, width=0.5)  # Adjust the bar width as desired
    ax2.set_ylabel('Numbers of participants')
    ax2.set_title('Participants Count Comparison')
    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')
    # Increase the y-axis limit
    ax2.set_ylim(top=max(y) * 1.1)
    # Adjust the layout
    plt.tight_layout()
    # Display the combined chart
    plt.show()







import pandas as pd
import matplotlib.pyplot as plt

def plot_lengths_comments(df_list, names):
    lengths = [len(df) for df in df_list]

    # Sort the lengths and names in ascending order
    lengths, names = zip(*sorted(zip(lengths, names)))

    fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figure size as desired
    bars = ax.bar(names, lengths, width=0.5)  # Adjust the bar width as desired
    ax.set_xlabel('Topics')
    ax.set_ylabel('Numbers of recommendations')
    ax.set_title('Comparison about numbers of recommendations about topics')

    # Rotate the x-axis labels by 90 degrees
    plt.xticks(rotation=90)

    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

    # Increase the y-axis limit
    ax.set_ylim(top=max(lengths) * 1.1)

    # Adjust the layout
    plt.tight_layout()

    # Display the bar chart
    plt.show()

    
    
    
    
    
    
import pandas as pd
import matplotlib.pyplot as plt

def plot_numbers_surveyors(df_list, names):
    lengths = [df['Comment_ID'].str[:5].nunique() for df in df_list]

    # Sort the lengths and names in ascending order
    lengths, names = zip(*sorted(zip(lengths, names)))

    fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figure size as desired
    bars = ax.bar(names, lengths, width=0.5)  # Adjust the bar width as desired
    ax.set_xlabel('Topics')
    ax.set_ylabel('Numbers of surveyors')
    ax.set_title('Comparison about numbers of surveyors giving recommadation about topics')

    # Rotate the x-axis labels by 90 degrees
    plt.xticks(rotation=90)

    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

    # Increase the y-axis limit
    ax.set_ylim(top=max(lengths) * 1.1)

    # Adjust the layout
    plt.tight_layout()

    # Display the bar chart
    plt.show()
    


import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_wordcloud(df, column_name):
    # Create a copy of the dataframe
    feedback_df_ = df.copy()

    # Drop any row that contains missing values in the specified column
    feedback_df_.dropna(subset=[column_name], inplace=True)

    # Convert the column to string type
    feedback_df_[column_name] = feedback_df_[column_name].astype(str)

    # Creating the text variable
    text = " ".join(review for review in feedback_df_[column_name])

    # Creating word cloud with text as argument in .generate() method
    wordcloud = WordCloud(collocations=False, background_color='white').generate(text)

    # Display the generated Word Cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()




    
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

def create_word_frequency_table(recommendations_list):
    # Download stopwords if not already downloaded
    nltk.download('stopwords')

    # Download punkt tokenizer if not already downloaded
    nltk.download('punkt')

    # Define stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords and create frequency table
    word_freq = FreqDist()
    for text in recommendations_list:
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        word_freq.update(tokens)

    # Print frequency table
    print("Word Frequency Table:")
    for word, freq in word_freq.most_common():
        print(f"{word}: {freq}")

    






    
