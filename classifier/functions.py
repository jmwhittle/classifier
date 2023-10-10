import numpy as np
import pandas as pd
import scipy.stats as stats
# from jenkspy import JenksNaturalBreaks

from datetime import date, datetime, timedelta, time
from calendar import monthrange
import matplotlib.pyplot as plt


def t_tester(df, col, group):
    """
    """
    
    group1 = df.loc[df[f'{group}'] == 0][[f'{col}']]
    group2 = df.loc[df[f'{group}'] == 1][[f'{col}']]
    
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')
    
    # Print the results
    print("##################")
    print(f"Sig test {col}")
    print("Group n:", group1.shape[0], ":", group2.shape[0])
    print("Group means: ", round(group1.mean()[0], 2), round(group2.mean()[0], 2))
    print("Group medians: ", round(group1.median()[0], 2), round(group2.median()[0], 2))
    print("Group Std Dev: ", round(np.std(group1)[0]), round(np.std(group2)[0]))
    print("T-Statistic:", t_statistic)
    print("P-Value:", p_value)
    

def days_to_end_of_quarter(date_input):
    """
    Input a date and get back an integer of the number of days till the end of quarter

    Args:
        date_input (str or datetime): date to use for determining the days left in quarter

    Returns:
        datetime of last day of quarter
    """
        
    return (get_last_day_of_quarter(date_input) - pd.to_datetime(date_input)).days

def get_last_day_of_quarter(date_input):
    """
    Input day of interest as string and get a datetime object back with the last day of the quarter

    Args:
        date_input (str or datetime): date to use for determining the last day of the quarter

    Returns:
        datetime of last day of quarter
    """
    quarter = pd.to_datetime(date_input).quarter
    year = pd.to_datetime(date_input).year
    last_month_of_quarter = 3 * quarter
    date_of_last_day_of_quarter = date(
        year, last_month_of_quarter, monthrange(year, last_month_of_quarter)[1]
    )

    return datetime.combine(date_of_last_day_of_quarter, datetime.min.time())


def ts_activity_plot(df, name:str, plot_name:str):
    df_gb = df.groupby(['day', 'email', name])['opp_id'].count().reset_index()
    df_gb = df_gb.groupby(['day', name])['opp_id'].mean().reset_index()
    df_gb = df_gb.pivot(index='day', columns=name, values='opp_id').reset_index().fillna(0)
    df_gb.columns = ['day', f'{name}_0', f'{name}_1']
    
    df_gb[f'{name}_0'] = df_gb[f'{name}_0'].rolling(7).mean()
    df_gb[f'{name}_1'] = df_gb[f'{name}_1'].rolling(7).mean()
    
    df_gb = df_gb.dropna()
    
    print("Missed mean: ", df_gb[f'{name}_0'].mean(), "Achieved mean: ", df_gb[f'{name}_1'].mean())
    
    # Plotting the data
    plt.figure(figsize=(10, 6))  # Optional: adjust the figure size
    plt.plot(df_gb['day'], df_gb[f'{name}_0'], label=f'{name}-missed')
    plt.plot(df_gb['day'], df_gb[f'{name}_1'], label=f'{name}-achived')
    plt.xlabel('Day')
    plt.ylabel('Value')
    plt.title(plot_name)  # Update the title with 'blast'
    plt.legend()
    plt.xticks(rotation=45)  # Optional: Rotate x-axis labels for better visibility
    plt.grid(True)  # Optional: Add gridlines
    plt.tight_layout()  # Optional: Improve spacing
    plt.show()
    
def eoq_activity_plot(df, name, plot_name):
    
    #df['days_to_eoq'] = df['day'].apply(days_to_end_of_quarter)
    
    df_gb = df.groupby(['day','days_to_eoq', 'email', name])['opp_id'].count().reset_index()
    df_gb = df_gb.groupby(['days_to_eoq', name])['opp_id'].mean().reset_index()
    df_gb = df_gb.pivot(index='days_to_eoq', columns=name, values='opp_id').reset_index().fillna(0)
    df_gb.columns = ['days_to_eoq', f'{name}_0', f'{name}_1']
    
    df_gb[f'{name}_0'] = df_gb[f'{name}_0'].rolling(7).mean()
    df_gb[f'{name}_1'] = df_gb[f'{name}_1'].rolling(7).mean()
    
    df_gb = df_gb.dropna()
    
    print("Missed mean: ", df_gb[f'{name}_0'].mean(), "Achieved mean: ", df_gb[f'{name}_1'].mean())
    
    # Plotting the data
    plt.figure(figsize=(10, 6))  # Optional: adjust the figure size
    plt.plot(df_gb['days_to_eoq'], df_gb[f'{name}_0'], label=f'{name}-missed')
    plt.plot(df_gb['days_to_eoq'], df_gb[f'{name}_1'], label=f'{name}-achived')
    plt.xlabel('Days to EOQ')
    plt.ylabel('Daily Average (7-day mvoing average)')
    plt.title(plot_name)  # Update the title with 'blast'
    plt.legend()
    plt.xticks(rotation=45)  # Optional: Rotate x-axis labels for better visibility
    plt.grid(True)  # Optional: Add gridlines
    plt.tight_layout()  # Optional: Improve spacing
    plt.gca().invert_xaxis()
    plt.show()
    
    
def eoq_activity_segment_plot(df, name, segmentor, plot_name):
    
    #df['days_to_eoq'] = df['day'].apply(days_to_end_of_quarter)
    
    df_gb = df.groupby(['day','days_to_eoq', 'email', segmentor, name])['opp_id'].count().reset_index()
    df_gb = df_gb.groupby(['days_to_eoq', segmentor, name])['opp_id'].mean().reset_index()
    df_gb = df_gb.pivot(index='days_to_eoq', columns=name, values='opp_id').reset_index().fillna(0)
    df_gb.columns = ['days_to_eoq', f'{name}_0', f'{name}_1']
    
    df_gb[f'{name}_0'] = df_gb[f'{name}_0'].rolling(7).mean()
    df_gb[f'{name}_1'] = df_gb[f'{name}_1'].rolling(7).mean()
    
    df_gb = df_gb.dropna()
    
    print("Missed mean: ", df_gb[f'{name}_0'].mean(), "Achieved mean: ", df_gb[f'{name}_1'].mean())
    
    # Plotting the data
    plt.figure(figsize=(10, 6))  # Optional: adjust the figure size
    plt.plot(df_gb['days_to_eoq'], df_gb[f'{name}_0'], label=f'{name}-missed')
    plt.plot(df_gb['days_to_eoq'], df_gb[f'{name}_1'], label=f'{name}-achived')
    plt.xlabel('Days to EOQ')
    plt.ylabel('Value')
    plt.title(plot_name)  # Update the title with 'blast'
    plt.legend()
    plt.xticks(rotation=45)  # Optional: Rotate x-axis labels for better visibility
    plt.grid(True)  # Optional: Add gridlines
    plt.tight_layout()  # Optional: Improve spacing
    plt.gca().invert_xaxis()
    plt.show()
    
    
def eoq_activity_segment_plot(df, name, segment, plot_name):
    
    #df['days_to_eoq'] = df['day'].apply(days_to_end_of_quarter)
    print(segment)
    df = df.loc[df['segment']==segment]
    
    df_gb = df.groupby(['day','days_to_eoq', 'email',  name])['opp_id'].count().reset_index()
    df_gb = df_gb.groupby(['days_to_eoq',  name])['opp_id'].mean().reset_index()
    df_gb = df_gb.pivot(index='days_to_eoq', columns=name, values='opp_id').reset_index().fillna(0)
    df_gb.columns = ['days_to_eoq', f'{name}_0', f'{name}_1']
    
    df_gb[f'{name}_0'] = df_gb[f'{name}_0'].rolling(7).mean()
    df_gb[f'{name}_1'] = df_gb[f'{name}_1'].rolling(7).mean()
    
    df_gb = df_gb.dropna()
    
    print("Missed mean: ", df_gb[f'{name}_0'].mean(), "Achieved mean: ", df_gb[f'{name}_1'].mean())
    
    # Plotting the data
    plt.figure(figsize=(10, 6))  # Optional: adjust the figure size
    plt.plot(df_gb['days_to_eoq'], df_gb[f'{name}_0'], label=f'{name}-missed')
    plt.plot(df_gb['days_to_eoq'], df_gb[f'{name}_1'], label=f'{name}-achived')
    plt.xlabel('Days to EOQ')
    plt.ylabel('Value')
    plt.title(f'{segment} {plot_name}')  # Update the title with 'blast'
    plt.legend()
    plt.xticks(rotation=45)  # Optional: Rotate x-axis labels for better visibility
    plt.grid(True)  # Optional: Add gridlines
    plt.tight_layout()  # Optional: Improve spacing
    plt.gca().invert_xaxis()
    plt.show()