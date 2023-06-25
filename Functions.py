import pandas as pd
import matplotlib.pyplot as plt

def count_missing_values_1(df):
    # Create a DataFrame to store the results
    missing_counts = pd.DataFrame(index=df.columns, columns=['Missing', 'Non-Missing'])

    # Loop over each column in the DataFrame
    for col in df.columns:
        # Count the number of missing and non-missing values for the column
        missing = df[col].isna().sum()
        non_missing = df[col].notna().sum()

        # Store the counts in the DataFrame
        missing_counts.loc[col, 'Missing'] = missing
        missing_counts.loc[col, 'Non-Missing'] = non_missing

    # Sort the DataFrame by the number of missing values
    missing_counts.sort_values(by=['Missing'], inplace=True, ascending=False)

    # Display the sorted results
    return missing_counts


def plot_comparison_2(df, new_df_column, df_column):
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
def split_sentences_3(df, column_name):
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



def plot_comparison_4(df, new_df_column, df_column):
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





def plot_observation_count_5(new_df, df_1, column_name):
    # Count observations in new_df_library with the condition
    new_df_count = new_df["Comment_ID"].str[:5].nunique()
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
    # Save the chart as an image
    plt.savefig("Chart of " + column_name + '.png', bbox_inches='tight')
    plt.show()






import matplotlib.pyplot as plt

def plot_lengths_comments_6(df_list, names):
    lengths = [len(df) for df in df_list]

    # Sort the lengths and names in ascending order
    lengths, names = zip(*sorted(zip(lengths, names)))

    fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figure size as desired
    bars = ax.bar(names, lengths, width=0.5)  # Adjust the bar width as desired
    ax.set_xlabel('Topics')
    ax.set_ylabel('Numbers of recommendations')
    ax.set_title('Numbers of recommendations by topics')

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

    # Save the chart as an image
    plt.savefig("Chart of " + "Numbers of recommendations by topics" + '.png', bbox_inches='tight')

    # Display the bar chart
    plt.show()

    
    
    
    

def plot_numbers_surveyors_7(df_list, names):
    lengths = [df['Comment_ID'].str[:5].nunique() for df in df_list]

    # Sort the lengths and names in ascending order
    lengths, names = zip(*sorted(zip(lengths, names)))

    fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figure size as desired
    bars = ax.bar(names, lengths, width=0.5)  # Adjust the bar width as desired
    ax.set_xlabel('Topics')
    ax.set_ylabel('Numbers of surveyors')
    ax.set_title('Numbers of surveyors giving recommadation by topics')

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
    
    # Save the chart as an image
    plt.savefig("Chart of " + "Numbers of surveyors giving recommadation by topics" + '.png', bbox_inches='tight')
    
    # Display the bar chart
    plt.show()
    


from wordcloud import WordCloud

def generate_wordcloud_8(df, column_name):
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

def create_word_frequency_table_9(recommendations_list):
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

    

    

    
    
    

def generate_response_counts_10(df_selected):
    # Create an empty dictionary to hold the counts for each response
    counts_dict = {}

    for col in df_selected.columns:
        # Get the value counts for that column and add it to the dictionary
        counts_dict[col] = df_selected[col].value_counts(dropna=False)

    # Create a new dataframe from the dictionary
    df_counts = pd.DataFrame.from_dict(counts_dict)

    # Fill any missing values with 0
    df_counts.fillna(0, inplace=True)

    # Transpose the dataframe so that the proposals are rows and the response counts are columns
    df_counts = df_counts.transpose()

    # Rename the index column to 'Proposal'
    df_counts.index.name = 'Questions'

    return df_counts
    
    
    
    
    
def plot_frequency_distribution_11(df, column, chart_name):
    # Get the frequency distribution of each category
    category_counts = df[column].value_counts()

    # Create a figure with proper layout
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.tight_layout(pad=5.0)

    # Create a bar chart of the category counts
    ax1.bar(category_counts.index, category_counts.values)
    ax1.set_xlabel('Frequency of visits')
    ax1.set_ylabel('Number of respondents')
    ax1.set_title(chart_name)

    # Add numbers for each bar
    for i, v in enumerate(category_counts.values):
        ax1.text(i, v+3, str(v), ha='center')

    # Rotate x-axis labels
    ax1.tick_params(axis='x', rotation=90)

    # Create a pie chart of the category percentages
    wedges, _, autotexts = ax2.pie(category_counts.values, labels=None, autopct='%1.1f%%', startangle=90)
    ax2.set_title(chart_name)
    ax2.legend(wedges, category_counts.index, title='Categories', loc='center left', bbox_to_anchor=(1, 0.5))

    # Set the chart axis labels
    ax2.set_xlabel('Percentage of respondents')

    # Increase the size of the y-axis
    ax1.set_ylim(0, category_counts.values.max() * 1.1)

    # Add percentage values to the pie chart
    for autotext in autotexts:
        autotext.set_color('white')

    # Save the chart as an image
    plt.savefig("Chart of " + chart_name + '.png', bbox_inches='tight')

    # Display the chart
    plt.show()







    
def generate_stacked_horizontal_bar_plot_12(dataframe, title, x_label, y_label):
    num_columns = len(dataframe.columns)
    num_bars = len(dataframe.index)
    figsize_height = num_bars *0.5

    # Create the bar plot with default width bars
    fig, ax = plt.subplots(figsize=(10, figsize_height))
    dataframe.plot.barh(stacked=True, ax=ax)

    # Set the title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Move the legend below the plot
    legend_position = 'lower center'
    if figsize_height < 3:
        legend_position = 'center'
    
    legend_labels = ax.legend(loc=legend_position, bbox_to_anchor=(0.5, -3/num_bars), ncol=num_columns).get_texts()
    ncol = len(legend_labels) // 2
    ax.legend(loc=legend_position, bbox_to_anchor=(0.5, -2.75/num_bars), ncol=ncol)

    # Add labels to the bars
    for patch in ax.patches:
        # Get the width of the patch
        width = patch.get_width()
        # Get the height of the patch
        height = patch.get_height()
        # Get the x position of the left edge of the patch
        x = patch.get_x()
        # Get the y position of the bottom edge of the patch
        y = patch.get_y()
        # Add the label to the patch
        ax.text(x + width / 2, y + height / 2, str(int(width)), ha='center', va='center', color='white')
    # Save the chart as an image
    plt.savefig("Chart of " + title + '.png', bbox_inches='tight')

    # Show the plot
    plt.show()

    
    
    
    
    
def generate_horizontal_bar_chart_13(dataframe, column_name, chart_title, x_label, y_label, color):
    # Create a dataframe with the specified column
    df = dataframe[[column_name]].copy()

    # Sort the dataframe by the specified column in descending order
    df.sort_values(by=column_name, ascending=False, inplace=True)

    # Create the horizontal bar chart
    ax = df.plot.barh(color=color, legend=False)

    # Set the title and labels
    ax.set_title(chart_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Add labels to the bars
    for i, v in enumerate(df[column_name]):
        ax.text(v + 1, i - 0.2, str(v), color='black')

    # Set the x-axis limits
    ax.set_xlim([0, max(df[column_name]) * 1.1])
    # Save the chart as an image
    plt.savefig("Chart of " + chart_title + '.png', bbox_inches='tight')

    # Show the plot
    plt.show()
    
    
    
    
    
    
    
    
    
    