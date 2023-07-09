#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


import pandas as pd
import numpy as np
import panel as pn
pn.extension('tabulator')

import hvplot.pandas


# In[3]:


# df = pd.read_csv('C:\\Users\\hp\\OneDrive\\Desktop\Result_Visualization.csv')


# In[3]:


# cache data to improve dashboard performance
if 'data' not in pn.state.cache.keys():

    # df = pd.read_csv('C:\\Users\\ratul\\OneDrive\\Documents\\Result_Visualization.csv')
    url = 'https://raw.githubusercontent.com/KHSakib/SIDVis/master/src/data/Result_Visualization.csv'
    df = pd.read_csv(url)

    pn.state.cache['data'] = df.copy()

else: 

    df = pn.state.cache['data']


# In[4]:



# In[5]:


# Make DataFrame Pipeline Interactive
idf = df.interactive()


# In[6]:


# Define Panel widgets
year_slider = pn.widgets.IntSlider(name='Accuracy', start=60, end=100, step=2, value=85)
year_slider


# In[7]:


# Radio buttons for CO2 measures
yaxis_co2 = pn.widgets.RadioButtonGroup(
    name='Y axis', 
    options=['Precision', 'Recall',],
    button_type='success'
)


# In[8]:


models = ['BERT', 'LSTM', 'BiLSTM', 'RF', 'SVM', 'GaussianNB', 'LR', 'KNeighbors']

co2_pipeline = (
    idf[
        (idf.Accuracy <= year_slider) &
        (idf.Models.isin(models))
    ]
    .groupby(['Models', 'Feature extraction', 'Performance', 'Accuracy', 'AUC of ROC'])[yaxis_co2].mean()
    .to_frame()
    .reset_index()    .sort_values(by='Accuracy')  
    .reset_index(drop=True)
)


# In[9]:




# In[10]:


co2_plot = co2_pipeline.hvplot(x = 'Accuracy', by='Models', y=yaxis_co2,line_width=2, title="Result Visualization")
co2_plot


# In[11]:


co2_table = co2_pipeline.pipe(pn.widgets.Tabulator, pagination='remote', page_size = 10, sizing_mode='stretch_width') 
co2_table


# In[12]:





# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
pn.extension('matplotlib')

# create example data
data = {'Models': ['BERT', 'LSTM', 'BiLSTM', 'RF', 'SVM', 'GaussianNB', 'LR', 'KNeighbors', 'RF', 'SVM', 'GaussianNB', 'LR', 'KNeighbors'],
        'Feature Extraction': ['word embedding', 'word embedding', 'word embedding', 'LIWC', 'LIWC', 'LIWC', 'LIWC', 'LIWC', 'TF-idf', 'TF-idf', 'TF-idf', 'TF-idf', 'TF-idf'],
        'accuracy': [0.88, 0.99, 0.99, 0.99, 0.91, 0.77, 0.78, 0.78, 0.99, 0.90, 0.84, 0.89, 0.87]}
df1 = pd.DataFrame(data)

# create a seaborn horizontal bar plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=df1, x='accuracy', y='Models', hue='Feature Extraction', orient='h')

# add labels and title
plt.xlabel('Accuracy')
plt.ylabel('Models')
plt.title('(B) Accuracy Based on Feature Extraction')

# create a panel with the plot
plot_pane = pn.pane.Matplotlib(plt.gcf(), dpi=144)


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import panel as pn
pn.extension('matplotlib')

# create example data
data = {'Models': ['BERT', 'LSTM', 'BiLSTM', 'RF', 'SVM', 'GaussianNB', 'LR', 'KNeighbors', 'BERT', 'LSTM', 'BiLSTM', 'RF', 'SVM', 'GaussianNB', 'LR', 'KNeighbors'],
        'Performance': ['Training', 'Training', 'Training', 'Training', 'Training', 'Training', 'Training', 'Training', 'Testing', 'Testing', 'Testing', 'Testing', 'Testing', 'Testing', 'Testing', 'Testing'],
        'accuracy': [0.88, 0.99, 0.99, 0.99, 0.91, 0.77, 0.78, 0.78, 0.88, 0.89, 0.88, 0.89, 0.88, 0.77, 0.78, 0.75]}
df2 = pd.DataFrame(data)

# create a seaborn vertical bar plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=df2, x='Models', y='accuracy', hue='Performance', orient='v', palette = "Blues")

# add labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('(D) Trainning and Testing accuracy')

# create a panel with the plot
plot_pane4 = pn.pane.Matplotlib(plt.gcf(), dpi=144)


# In[16]:


# import plotly.graph_objects as go

# # create example data
# data = {'Models': ['BERT', 'LSTM', 'BiLSTM', 'RF', 'SVM', 'GaussianNB', 'LR', 'KNeighbors', 'RF', 'SVM', 'GaussianNB', 'LR', 'KNeighbors'],
#         'Feature Extraction': ['word embedding', 'word embedding', 'word embedding', 'LIWC', 'LIWC', 'LIWC', 'LIWC', 'LIWC', 'TF-idf', 'TF-idf', 'TF-idf', 'TF-idf', 'TF-idf'],
#         'accuracy': [0.88, 0.99, 0.99, 1.0, 0.87, 0.77, 0.78, 0.68, 0.99, 0.95, 0.84, 0.89, 0.79]}
# df1 = pd.DataFrame(data)

# labels = df1['Models']
# values = df1['accuracy']

# fig = go.Figure(data=[
#         go.Pie(labels=labels, values=values)
#     ])

# # displaying the Pie Chart
# fig.show()


# In[17]:


# import panel as pn
# import matplotlib.pyplot as plt

# pn.extension('matplotlib')

# # create a list of accuracy values
# accuracy = [98.85, 99.00, 99.42, 99.71, 99.85, 99.85, 99.85, 99.85, 99.85, 99.85]
# accuracy1 = [50.14, 93.71, 93.85, 94.71, 95, 94.85, 94.85, 95.00, 95.00, 95.00]
# accuracy2 = [60.14, 67.71, 83.85, 91.71, 95, 94.85, 94.85, 95.00, 96.00, 99.00]
# accuracy3 = [75.14, 93.71, 94.85, 94.71, 95, 96.85, 97.85, 97.00, 98.00, 99.00]
# accuracy4 = [60.14, 70.71, 77.85, 85.71, 91, 94.85, 94.85, 95.00, 95.40, 95.00]
# accuracy5 = [55.14, 55.71, 60.85, 60.85, 60.85, 60.85, 62, 65.00, 65.00, 65.00]
# accuracy6 = [50.14, 73.71, 73.85, 74.71, 75, 74.85, 74.85, 75.00, 75.00, 75.00]
# accuracy7 = [70.14, 73.71, 73.85, 84.71, 85, 84.85, 84.85, 85.00, 85.00, 85.00]
# # create a list of epoch values
# epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# # Define a function to plot the data
# def plot_accuracy(epochs, accuracy, accuracy1):
#     fig, ax = plt.subplots()
#     ax.plot(epochs, accuracy, label='BERT')
#     ax.plot(epochs, accuracy1, label='LSTM')
#     ax.plot(epochs, accuracy2, label='BiLSTM')
#     ax.plot(epochs, accuracy3, label='RF')
#     ax.plot(epochs, accuracy4, label='SVM')
#     ax.plot(epochs, accuracy5, label='GaussianNB')
#     ax.plot(epochs, accuracy6, label='LR')
#     ax.plot(epochs, accuracy7, label='KNeighbors')
#     ax.set_xlabel('Epochs')
#     ax.set_ylabel('Accuracy')
#     ax.set_title('Accuracy over Epochs')
#     plt.figure(figsize=(20, 18))
#     ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1))
#     return fig


# # create a slider widget for selecting the epoch
# epoch_slider = pn.widgets.IntSlider(start=1, end=10, value=1, name='epochs')
# # create a Matplotlib pane
# plot_pane1 = pn.pane.Matplotlib(plot_accuracy(epochs, accuracy, accuracy1))

# # set the width of the pane
# plot_pane1.width = 800

# # create a panel dashboard
# dashboard = pn.Column(
#     '# Accuracy over Epochs',
#     pn.Row(
#         pn.Column(plot_pane1, width=800),
#     ),
# )

# # Display the dashboard
# dashboard.servable()


# In[18]:


import panel as pn
import matplotlib.pyplot as plt

pn.extension('matplotlib')

# create a dictionary of accuracy values for 5 models
accuracy = {
    'BERT': [77.82, 83.54, 84.69, 85.62, 86.28, 86.57, 86.81, 87.14, 87.14, 87.47],
       'LSTM': [84.41, 91.17, 93.39, 94.93, 96.26, 96.87, 97.74, 98.01, 98.19, 98.63],
       'BiLSTM': [85.52, 91.76, 94.22, 95.76, 96.96, 97.46, 97.63, 97.66, 98.19, 98.52],
       'RF_TF-idf': [93.25, 92.94, 97.16, 96.53, 98.17, 97.84, 98.56, 98.34, 98.95, 98.64],
       'SVM_TF-idf': [55.66, 55.66, 85.38, 88.73, 89.78, 90.03, 89.95, 90.02, 90.02, 89.99],
       'GaussianNB_TF-idf': [84.05, 84.05, 84.05, 84.05, 84.05, 84.05, 84.05, 84.05, 84.05, 84.05],
       'LR_TF-idf': [55.66, 55.66, 84.20, 87.16, 88.58, 89.21, 89.18, 89.11, 89.11, 89.11],
       'KNeighbors_TF-idf': [98.03, 92.63, 91.45, 89.82, 89.66, 88.97, 87.99, 87.63, 87.26, 86.84],
       'RF_LIWC': [92.02, 93.09, 97.08, 96.73, 98.33, 97.93, 98.77, 98.55, 99.13, 98.99],
       'SVM_LIWC': [56.07, 56.07, 78.82, 83.27, 86.14, 88.54, 92.14, 96.25, 99.24, 99.86],
       'GaussianNB_LIWC': [76.95, 76.95, 76.95, 76.95, 76.95, 76.95, 76.95, 76.95, 76.95, 76.95],
       'LR_LIWC': [78.58, 79.34, 79.24, 77.48, 77.39, 78.14, 78.03, 77.26, 77.46, 78.20],
       'KNeighbors_LIWC': [99.86, 95.64, 85.27, 86.67, 81.63, 83.18, 79.75, 81.00, 78.65, 79.87]
}

# Define a function to plot the data
def plot_accuracy(model, epoch):
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), accuracy[model])
    ax.plot(epoch, accuracy[model][epoch-1], 'ro')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'(A) {model} Accuracy over Epochs')
    plt.figure(figsize=(20, 18))
    return fig

# create a dropdown widget for selecting the model
#model_widget = pn.widgets.Select(options=list(accuracy.keys()), name='Model')

# create a slider widget for selecting the epoch
epoch_slider = pn.widgets.IntSlider(start=1, end=10, value=1, name='Epoch')

# create an interactive plot using the interact function
interactive_plot = pn.interact(plot_accuracy, model=list(accuracy.keys()), epoch=epoch_slider)

# Define the panel dashboard
dashboard = pn.Column(
    '# Accuracy over Epochs',
    interactive_plot
)

# Display the dashboard
#dashboard.servable()


# In[19]:


#Layout using Template
template = pn.template.FastListTemplate(
    title='Suicide Ideation Detection Result Visualization', 
    sidebar=[pn.pane.Markdown("## SIDVis Details"), 
             pn.pane.Markdown("#### Nowadays, suicide is a critical concern, with mental illness, substance abuse, financial stress, and traumatic experiences standing out as the primary reasons behind suicidal activities. Detecting suicidal ideation (SID) is an intricate undertaking, as it entails considering multiple factors that rely on the language associated with suicide. Therefore, SIDVis aims to analyze social media texts and encompasses the performance of various machine learning (ML) and deep learning (DL) techniques for identifying suicidal ideation.") 
            # pn.pane.PNG('climate_day.png', sizing_mode='scale_both'),
             ],
    main=[pn.Row(pn.Column(pn.Column(interactive_plot, width=800)), 
                 pn.Column(plot_pane)), 
          pn.Row(pn.Column(yaxis_co2,pn.pane.Markdown("### (C) Result Table"),year_slider, co2_table.panel(width=800)),
                pn.Column(plot_pane4)
                )],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)


# In[20]:


# template.show()
# template.servable()
pn.serve(template)

# from flask import Flask


# app = Flask(__name__)


# if __name__ == '__main__':
#     pn.serve(template, 8080)
