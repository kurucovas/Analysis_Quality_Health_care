import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings

def main():
    line = ("="*65)
    print("QUALITY OF HEALTH CARE PROVIDED TO ONCOLOGY PATIENTS")
    print(line)
    print()
    print("""    The purpose of data analysis is to evaluate the quality of health care provided to oncology patients
    during years 2019 – 2022 at the health care provider, which is MO Institute. 
    The quality of health care is evaluated based on the following indicators:  
    - 30-day mortality after the end of systemic treatment, i.e. the number of patients died within 30 days of the last treatment application
    - occurrence of febrile neutropenia in patients on systemic treatment, i. e.  how often this condition occurs in treated patients.""" )
    print()
    print("""    The analyzed population sample
    The analysis is related to patients, who were diagnosed with 1 of the most frequently occuring oncological diagnoses during the years 2019-2022: 
    •	breast cancer,  
    •	lung cancer,
    •	colon cancer.
    The research sample consisted of 2 000 patients of MO Institute. """ )
    print()
    print("1.indicator: 30-day mortality after the end of systemic treatment")
    print(line)
    print()

    # to read datasets from csv files by using library pandas (pandas as pd), df = dataframe = table, we create 3 objects:
    data_patients_df = pd.read_csv('my_environment1/MO_data_patients.csv', delimiter=';')
    data_treatment_df = pd.read_csv('my_environment1/MO_data_treatment.csv', delimiter=';')
    data_neutropenia_df = pd.read_csv('my_environment1/MO_data_neutropenia.csv', delimiter=';')

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    data_patients_df = create_column_diagnosis_name(data_patients_df)   # Apply the function to create the 'diagnosis_name' column
    data_patients_df = create_column_gender_m_w(data_patients_df)
    data_patients_df = create_column_year_of_death(data_patients_df)
    data_patients_df = create_column_age_structure_at_the_time_of_death(data_patients_df)
    data_patients_df = create_column_number_of_patients_treated_in_2019(data_patients_df)  # Apply the function to update the DataFrame
    data_patients_df = create_column_date_of_last_treatment_application(data_patients_df, data_treatment_df)    
    data_patients_df = create_column_number_of_patients_treated_in_2020(data_patients_df)  
    data_patients_df = create_column_number_of_patients_treated_in_2021(data_patients_df)   
    data_patients_df = create_column_number_of_patients_treated_in_2022(data_patients_df)
    data_patients_df = create_column_mortalita(data_patients_df)
    data_patients_df = create_column_30_days_mortality_count(data_patients_df)
    
    formatted_table = create_Tab_nb_1_Cancer_Development_of_30_days_mortality_in_2019_2022(data_patients_df)
    print("Tab_nb_1_Cancer_Development_of_30_days_mortality_in_2019_2022:")
    print(formatted_table)
    print("During the period of 2019 - 2022, 30-day mortality occurred in total in 79 cases (see tab. nb. 1)")
    print()
    print("""30-day mortality by diagnosis:
    - breast cancer – the highest number of cases (39), of which most cases were recorded in women over the age of 50 (22);
    - lung cancer – total of 21 registered cases, of which men over the age of 50 have the highest representation in a total of 18 cases;
    - colon cancer – total of 19 recorded cases, of which men over the age of 50 again represented the highest count in a total of 11 cases.""" )
    print()
    print("Development of 30-day mortality in years is displayed on the graph nb. 1")
    print()
    print("""From the point of view of age group at the time of death, the 30-day mortality most often occurred at age group 50-65 years, see chart nb. 2.""")
    print()
    print("""From the point of view of the clinical stage, the 30-day mortality increases with increasing clinical stage, see regression analyse in chart nb. 3.""")
    print()
    
    Graph_nb_1_Cancer_Development_of_30_days_mortality_in_2019_2022(data_patients_df)
    Graph_nb_2_30_days_mortality_by_age_group(data_patients_df)
    Graph_nb_3_REGRESSION_ANALYSE_30_days_mortality_by_clinical_stage(data_patients_df)
    
    contingency_table_2 = create_Tab_nb_2_Number_of_treated_patients_in_2019_2022(data_patients_df)
    print(f'Tab_nb_2_Number_of_treated_patients_in_2019_2022\n{contingency_table_2}')
    
    combined_df = create_Tab_nb_3_Cancer_30_days_mortality_2019_2022(data_patients_df)
    print("\nTab_nb_3_Cancer 30 days mortality_2019-2022_in absolute and percentage terms:")
    print(combined_df)
    print()
    print("30-day mortality in percentage terms")
    print()
    print("""    In relation to the total number of treated patients for each diagnosis in individual years, 30-days mortality dominated 
    in lung cancer (see tab. nb. 3 and graph nb. 4), in 2019 it was at the level of 11.9%, it has been decreasing since this year and
    in 2022 it was at the level of 5%. 
    For the other monitored diagnoses, 30-days mortality not exceeding the level of 2.5% was recorded in the last 2 reporting years.""")

    Graph_nb_4_development_30_days_mortality_in_percentage()
    
    line2 = "="*80
    print()
    print("2.indicator: Occurrence of febrile neutropenia in patients on systemic treatment")
    print(line2)
    print()
    
    ids_with_multiple_dates = check_multiple_dates_neutropenia(data_neutropenia_df)
    if not ids_with_multiple_dates.empty:
        print(f'Found IDs with multiple dates: {ids_with_multiple_dates}')
    else:
        print("All ID numbers are tied to only 1 date of neutropenia, what means, neutropenia occurred in patients only once.") 

    data_patients_df = merge_and_process_date_neutropenia(data_patients_df, data_neutropenia_df)

    warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

    contingency_table_4 = create_Tab_nb_4_Occurence_of_neutropenia_in_absolute_and_percentage_terms(data_patients_df)

    # Calculate total sum of the 'Neutropenia_count' column
    total_neutropenia = contingency_table_4['Neutropenia_COUNT'].sum()
    total_number_of_patients_per_period_2019_2022 = 2000
    percentage_neutropenia = total_neutropenia/total_number_of_patients_per_period_2019_2022*100

    print()
    print(f'Analysis in terms of the total number of patients for the period 2019-2022:\nThe total number of neutropenia cases during the monitored period is {total_neutropenia} from the total number of {total_number_of_patients_per_period_2019_2022} patients, \nwhat represents the occurrence of neutropenia at the level {percentage_neutropenia:.2f}%.')
    print()
    print("""In relation to the total number of patients treated for the relevant diagnoses in individual years the highest rate of occurrence \nof febrile neutropenia was recorded in year 2019 in lung cancer, at the level of 10.2%, while in 2022 it fell to 5.7%.\nIn the year-on-year comparison, decreasing trend is registered also in the case of occurence of febrile neutropenia in breast cancer.\nIn case of colon cancer, the rate of febrile neutropenia increased slightly, from 2,8% in 2021 to 5,6% in 2022 (see tab. nb. 4).""")
    print()
    print('Tab_nb_4_Occurence_of_febrile_neutropenia_in_absolute_and_percentage_terms:')
    print(tabulate(contingency_table_4, headers='keys', tablefmt='pretty', showindex=False))  # to display contingency table in table format using library tabulate

    final_table = find_closest_dates(data_patients_df, data_treatment_df)
    print() 
    print(f'Tab_nb_5_Occurence of febrile neutropenia by application order\n {final_table}')
    
    result_df = find_most_frequent_values(final_table)
    print()
    print('Tab_nb_6_Incidence of febrile neutropenia by application order')
    print(tabulate(result_df, headers='keys', tablefmt='pretty', showindex=False))

    Graph_nb_5_Incidence_of_Febrile_Neutropenia_by_Application_Order(result_df)

    print()
    print("""From the point of view of treatment application order, febrile neutropenia occurred most often after the 2nd treatment application, see tab. nb. 6 and graph nb. 5.""")
    print()
    print("CONCLUSION:")
    print("""    Based on monitored indicators of health care provision for oncology patients we can evaluate high quality of health care provision,
    as low 30-day mortality is reported, as well as a non-significant rate occurrence of febrile neutropenia.
    In a year-on-year comparison, the monitored indicators point to a constant improvement in the quality of provided health care,
    which was reflected in a decrease in 30-day mortality, as well as in a decrease in the incidence of febrile neutropenia.""")

    # Call plt.show() to prevent the script from closing the windows with graphs
    plt.show()                                                      # This blocks the script and keeps the windows open after execution

def create_column_diagnosis_name(data_patients_df):
    # Function to assign 'diagnosis name' based on 'diagnosis code'
    data_patients_df['diagnosis_name'] = data_patients_df['diagnosis'].apply(
        lambda x: 'breast cancer' if isinstance(x, str) and x.startswith('C50') else
        ('lung cancer' if isinstance(x, str) and x.startswith('C34') else
         ('colon cancer' if isinstance(x, str) and x.startswith(('C20', 'C19', 'C18')) else np.nan)))
    return data_patients_df                                         # Return the updated data_patients_df

def create_column_gender_m_w(data_patients_df):
    # Function to assign 'gender' based on 'gender codes'
    data_patients_df['gender_m_w'] = data_patients_df['gender'].apply(
        lambda x: 'man' if x == 1 else ('woman' if x == 2 else np.nan))
    return data_patients_df  

def create_column_year_of_death(data_patients_df):
    data_patients_df['year_of_death'] = data_patients_df['date_of_death'].apply(
        lambda x: pd.to_datetime(x, format='%d.%m.%Y', errors='coerce').year if pd.notna(x) else np.nan)
    # Convert 'year_of_death' to integers to remove decimal part
    data_patients_df['year_of_death'] = data_patients_df['year_of_death'].astype('Int64')  # Using 'Int64' to handle NaT (missing values) properly
    return data_patients_df  

def create_column_age_structure_at_the_time_of_death(data_patients_df):
    data_patients_df['age_structure_at_the_time_of_death'] = data_patients_df.apply(
        lambda row: "0-35" if pd.notna(row['year_of_death']) and pd.notna(row['year_of_birth']) and (row['year_of_death'] - row['year_of_birth']) < 35 else
        ("35-49" if pd.notna(row['year_of_death']) and pd.notna(row['year_of_birth']) and (row['year_of_death'] - row['year_of_birth']) <= 49 else
        ("50-65" if pd.notna(row['year_of_death']) and pd.notna(row['year_of_birth']) and (row['year_of_death'] - row['year_of_birth']) <= 65 else "65+")), axis=1)
    return data_patients_df  

def create_column_number_of_patients_treated_in_2019(data_patients_df):
    # Convert 'date_of_diagnosis' to datetime format
    data_patients_df['date_of_diagnosis'] = pd.to_datetime(data_patients_df['date_of_diagnosis'], format='%d.%m.%Y', errors='coerce')
    # Apply condition 1: If 'date_of_diagnosis' year is 2019, set 'number_of_patients_treated_in_2019' to 1
    data_patients_df['number_of_patients_treated_in_2019'] = data_patients_df['date_of_diagnosis'].apply(
        lambda x: 1 if pd.notnull(x) and x.year == 2019 else 0)
    return data_patients_df                                                           

def create_column_date_of_last_treatment_application(data_patients_df, data_treatment_df):
    data_treatment_df['date_of_treatment_application'] = pd.to_datetime(data_treatment_df['date_of_treatment_application'], errors='coerce')
    latest_dates = data_treatment_df.groupby('id')['date_of_treatment_application'].max().reset_index() # Get the latest 'date_of_treatment_application' for each 'id'
    data_patients_df = data_patients_df.merge(latest_dates, on='id', how='left')
    data_patients_df.rename(columns={'date_of_treatment_application': 'date_of_last_treatment_application'}, inplace=True) # Rename column to 'date_of_last_treatment_application'
    return data_patients_df                                                          
                                                     
def create_column_number_of_patients_treated_in_2020(data_patients_df):
    data_patients_df['number_of_patients_treated_in_2020'] = data_patients_df['date_of_diagnosis'].apply(lambda x: 1 if pd.to_datetime(x, errors='coerce').year == 2020 else 0)
    # Apply condition 2: If 'number_of_patients_treated_in_2019' is 1 and year in 'date_of_last_treatment_application' is 2020, 2021, or 2022
    condition_2 = (data_patients_df['number_of_patients_treated_in_2019'] == 1) & \
                  (data_patients_df['date_of_last_treatment_application'].apply(lambda x: pd.to_datetime(x, errors='coerce').year in [2020, 2021, 2022]))
    data_patients_df.loc[condition_2, 'number_of_patients_treated_in_2020'] = 1  # If condition 2 is true, set 'number_of_patients_treated_in_2020' to 1
    return data_patients_df                                                      # Return the updated DataFrame

def create_column_number_of_patients_treated_in_2021(data_patients_df):
    # Apply condition 1: Check if year in 'date_of_diagnosis' is 2021
    data_patients_df['number_of_patients_treated_in_2021'] = data_patients_df['date_of_diagnosis'].apply(lambda x: 1 if pd.to_datetime(x, errors='coerce').year == 2021 else 0)
    # Apply condition 2: If 'number_of_patients_treated_in_2020' is 1 and year in 'date_of_last_treatment_application' is 2021 or 2022
    condition_2 = (data_patients_df['number_of_patients_treated_in_2020'] == 1) & \
                  (data_patients_df['date_of_last_treatment_application'].apply(lambda x: pd.to_datetime(x, errors='coerce').year in [2021, 2022]))
    data_patients_df.loc[condition_2, 'number_of_patients_treated_in_2021'] = 1  # If condition 2 is true, set 'number_of_patients_treated_in_2021' to 1
    return data_patients_df                                                      

def create_column_number_of_patients_treated_in_2022(data_patients_df):
    data_patients_df['number_of_patients_treated_in_2022'] = data_patients_df['date_of_diagnosis'].apply(lambda x: 1 if pd.to_datetime(x, errors='coerce').year == 2022 else 0)
    condition_2 = (data_patients_df['number_of_patients_treated_in_2021'] == 1) & \
                  (data_patients_df['date_of_last_treatment_application'].apply(lambda x: pd.to_datetime(x, errors='coerce').year == 2022))
    data_patients_df.loc[condition_2, 'number_of_patients_treated_in_2022'] = 1 
    return data_patients_df                                                     

def create_column_mortalita(data_patients_df):
    # Ensure both 'date_of_death' and 'date_of_last_treatment' are in datetime format
    data_patients_df['date_of_death'] = pd.to_datetime(data_patients_df['date_of_death'], format='%d.%m.%Y', errors='coerce')
    data_patients_df['date_of_last_treatment_application'] = pd.to_datetime(data_patients_df['date_of_last_treatment_application'], errors='coerce')
    # Apply the calculation directly using lambda function
    data_patients_df['mortalita'] = data_patients_df.apply(
        lambda row: "30_days_mortality" 
        if pd.notna(row['date_of_death']) and pd.notna(row['date_of_last_treatment_application']) 
        and 0 < (row['date_of_death'] - row['date_of_last_treatment_application']).days <= 30
        else None, axis=1)
    return data_patients_df

def create_column_30_days_mortality_count(data_patients_df):
    data_patients_df['date_of_death'] = pd.to_datetime(data_patients_df['date_of_death'], errors='coerce')
    data_patients_df['date_of_last_treatment_application'] = pd.to_datetime(data_patients_df['date_of_last_treatment_application'], errors='coerce')
    # Create the column '30_days_mortality_count' with 1 if 'mortalita' contains '30-days_mortality', else 0
    data_patients_df['30_days_mortality_count'] = data_patients_df['mortalita'].apply(lambda x: 1 if x == '30_days_mortality' else 0)
    # Convert the column to integer, which will remove the decimal (and handle NaN values)
    data_patients_df['30_days_mortality_count'] = data_patients_df['30_days_mortality_count'].astype(int)
    return data_patients_df

def create_Tab_nb_1_Cancer_Development_of_30_days_mortality_in_2019_2022(data_patients_df):
    # Generate contingency table
    contingency_table = pd.crosstab(
        index=[data_patients_df['diagnosis_name'], data_patients_df['year_of_death']],
        columns=[data_patients_df['age_structure_at_the_time_of_death'], data_patients_df['gender_m_w']],
        values=data_patients_df['30_days_mortality_count'],
        aggfunc='sum',
        margins=True)
    contingency_table = contingency_table.fillna(0)                              # Replace NaN values with 0
    # Convert all numeric values to integers
    contingency_table = contingency_table.applymap(lambda x: int(x) if isinstance(x, (int, float)) else x)
    contingency_table_with_subtotals = pd.DataFrame()                            
    # Loop through each unique diagnosis and add rows for each year, then subtotal
    for diagnosis in contingency_table.index.get_level_values(0).unique():   
        diagnosis_group = contingency_table.xs(diagnosis, level=0)               # Filter table for this diagnosis group  
        subtotal_row = diagnosis_group.sum(axis=0).rename(('Total', diagnosis))  # Add subtotal row (sum of the years)
        # Append the rows for this diagnosis group and subtotal row
        contingency_table_with_subtotals = pd.concat([contingency_table_with_subtotals, diagnosis_group, subtotal_row.to_frame().T])
    # Drop the last subtotal row (marginal totals)
    rows = contingency_table_with_subtotals.index
    contingency_table_with_subtotals = contingency_table_with_subtotals.drop(rows[-2])
    # Reset index to move 'diagnosis_name' and 'year_of_death' to columns
    contingency_table_with_subtotals = contingency_table_with_subtotals.reset_index()
    # Rename 1. column, which was called automatically as'index' 
    contingency_table_with_subtotals.rename(columns={'index': 'diagnosis and years'}, inplace=True)
    # Convert the DataFrame to a pretty-printed table
    table = tabulate(contingency_table_with_subtotals, headers='keys', tablefmt='pretty', showindex=False, numalign="right", floatfmt="d")
    # Add a line before and after each subtotal row
    table_lines = table.splitlines()
    table_with_lines = []
    for line in table_lines:
        # Add line before subtotal
        if "Total" in line:                           # Detect subtotal rows by the presence of 'Total'
            table_with_lines.append('-' * len(line))  # Add a line before the subtotal
        table_with_lines.append(line)
        # Add line after subtotal
        if "Total" in line:
            table_with_lines.append('-' * len(line))  # Add a line after the subtotal
    table_with_lines = '\n'.join(table_with_lines)    # Join back all lines into a single string
    return table_with_lines

def Graph_nb_1_Cancer_Development_of_30_days_mortality_in_2019_2022(data_patients_df):
    # Group data by 'diagnosis_name' and 'year_of_death', and sum '30_days_mortality_count'
    grouped_data_1 = data_patients_df.groupby(['diagnosis_name', 'year_of_death'])['30_days_mortality_count'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    for diagnosis in grouped_data_1['diagnosis_name'].unique():
        diagnosis_data = grouped_data_1[grouped_data_1['diagnosis_name'] == diagnosis]
        plt.plot(diagnosis_data['year_of_death'], diagnosis_data['30_days_mortality_count'], label=diagnosis)
    # Customize the plot
    plt.gcf().set_facecolor('lightblue')  
    plt.grid(axis='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel('30-day mortality count')
    plt.title('Graph nb. 1_Cancer_Development_of_30_days_mortality_in_2019_2022')
    plt.xticks([2019, 2020, 2021, 2022])
    plt.yticks([0, 5, 10, 15, 20])
    plt.legend()                                                           
    plt.tight_layout()  
    plt.show(block=False)                            # Show the plot but do not block further execution

def Graph_nb_2_30_days_mortality_by_age_group(data_patients_df):
    age_order = ['0-35', '35-49', '50-65', '65+']
    # Convert 'age_structure_at_the_time_of_death' to categorical
    data_patients_df['age_structure_at_the_time_of_death'] = pd.Categorical(
        data_patients_df['age_structure_at_the_time_of_death'], categories=age_order, ordered=True)
    # Group data by diagnosis name and age structure at time of death
    grouped_data_2 = data_patients_df.groupby(['diagnosis_name', 'age_structure_at_the_time_of_death'])['30_days_mortality_count'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    for diagnosis in grouped_data_2['diagnosis_name'].unique():
        diagnosis_data = grouped_data_2[grouped_data_2['diagnosis_name'] == diagnosis]
        plt.plot(diagnosis_data['age_structure_at_the_time_of_death'], diagnosis_data['30_days_mortality_count'], label=diagnosis)
    plt.gcf().set_facecolor('lightblue') 
    plt.grid(axis='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)  
    plt.xlabel('Age Group at Time of Death')
    plt.ylabel('30-Day Mortality_COUNT')
    plt.title('Graph nb. 2_Cancer_30-Day Mortality in period of years 2019-2022 by Age Group at Time of Death')
    plt.xticks(age_order, rotation=45)                                             # Ensure correct order of age groups on the x-axis
    # Set y-axis ticks to integers
    ymin, ymax = plt.gca().get_ylim()
    yticks = range(int(ymin), int(ymax) + 1)
    plt.yticks(yticks)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

def Graph_nb_3_REGRESSION_ANALYSE_30_days_mortality_by_clinical_stage(data_patients_df):
    # Ensure 'clinical_stage' is treated as a categorical variable and encode it numerically
    clinical_stage_mapping = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    data_patients_df['clinical_stage_numeric'] = data_patients_df['clinical_stage'].map(clinical_stage_mapping)
    # Group data by diagnosis and clinical stage
    grouped_data_2 = data_patients_df.groupby(['diagnosis_name', 'clinical_stage_numeric'])['30_days_mortality_count'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    # Set a color palette for the different diagnoses
    palette = sns.color_palette("Set1", n_colors=len(grouped_data_2['diagnosis_name'].unique()))
    for i, diagnosis in enumerate(grouped_data_2['diagnosis_name'].unique()):
        diagnosis_data = grouped_data_2[grouped_data_2['diagnosis_name'] == diagnosis]
        # Plot the regression line using seaborn's regplot
        sns.regplot(x='clinical_stage_numeric', y='30_days_mortality_count', data=diagnosis_data, 
                    scatter=True, fit_reg=True, label=diagnosis, 
                    scatter_kws={'color': palette[i]},               # Set color for the scatter points (dots)
                    line_kws={"color": palette[i]})                  # Set color for the regression line
    # Use numpy.polyfit to fit a linear regression and get the slope and intercept
        slope, intercept = np.polyfit(diagnosis_data['clinical_stage_numeric'], diagnosis_data['30_days_mortality_count'], 1)
        # Create the equation string
        equation = f'y = {slope:.2f}x + {intercept:.2f}'
        plt.text(diagnosis_data['clinical_stage_numeric'].iloc[-1] + 0.1, diagnosis_data['30_days_mortality_count'].iloc[-1],
                 equation, color=palette[i], fontsize=12, ha='left', va='bottom')
    plt.gcf().set_facecolor('lightblue')  
    plt.grid(axis='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)  
    plt.xlabel('Clinical Stage')
    plt.ylabel('30-Day Mortality Count')
    plt.title('Graph nb. 3_REGRESSION_ANALYSE_Cancer_30-Day Mortality in years 2019-2022 by Clinical Stage')
    plt.xticks([1, 2, 3, 4], ['I', 'II', 'III', 'IV'])
    plt.xlim(1, 4)
    plt.yticks([0, 5, 10, 15, 20, 25])
    plt.legend(title="Diagnosis")
    plt.tight_layout()
    plt.show(block=False)

def create_Tab_nb_2_Number_of_treated_patients_in_2019_2022(data_patients_df):
    # Melt the dataframe to long format
    melted_df = data_patients_df.melt(
        id_vars=['diagnosis_name'],                                                        # 'diagnosis_name' as identifier column
        value_vars=['number_of_patients_treated_in_2019', 'number_of_patients_treated_in_2020', 
                    'number_of_patients_treated_in_2021', 'number_of_patients_treated_in_2022'],  # columns to aggregate
        var_name='year_of_treatment',                                                      # new column representing year
        value_name='number_of_patients_treated')                                           # column holding the count of treated patients
    # Create contingency table by summing the 'number_of_patients_treated' across diagnoses and years
    contingency_table_2 = pd.crosstab(
        index=[melted_df['diagnosis_name'], melted_df['year_of_treatment']],               # Rows: diagnosis and treatment years
        columns='Total',                                                                   # single column representing total count for each combination
        values=melted_df['number_of_patients_treated'],                                    # sum number of treated patients
        aggfunc='sum', )                                                                   # sum counts across the years
    return contingency_table_2

def create_Tab_nb_3_Cancer_30_days_mortality_2019_2022(data_patients_df):
    # Melt the dataframe to long format
    melted_df = data_patients_df.melt(
        id_vars=['diagnosis_name'],                                                        # 'diagnosis_name' as identifier column
        value_vars=['number_of_patients_treated_in_2019', 'number_of_patients_treated_in_2020', 
                    'number_of_patients_treated_in_2021', 'number_of_patients_treated_in_2022'],  # columns to aggregate
        var_name='year_of_treatment',                                                      # new column representing year
        value_name='number_of_patients_treated'                                            # column holding the count of treated patients
    )
    # Create contingency table by summing the 'number_of_patients_treated' across diagnoses and years
    contingency_table_3 = pd.crosstab(
        index=[melted_df['diagnosis_name'], melted_df['year_of_treatment']],              # Rows: diagnosis and treatment years
        columns='Total',                                                                  # single column representing total count for each combination
        values=melted_df['number_of_patients_treated'],                                   # sum number of treated patients
        aggfunc='sum',)                                                                   # sum counts across the years
    # Extract 'Total' column for the "Total" values and create the 'All' column 
    total_column_values = contingency_table_3['Total'].tolist()                           # Extract 'Total' column from contingency_table
    all_column_values = [7, 5, 19, 8, 0, 11, 3, 5, 7, 4, 3, 7]                            
    # Create a DataFrame with the 'Total' and 'All' columns
    combined_df = pd.DataFrame({                                                          # Create a DataFrame with 2 lists
        'Total': total_column_values,
        'All': all_column_values})
    # Calculate the 'Percentage' column
    combined_df['Percentage'] = (combined_df['All'] / combined_df['Total']) * 100         # Calculate percentage
    combined_df['Percentage'] = combined_df['Percentage'].round(1)                        # Round the 'Percentage' column to 1 decimal place
    # Define the diagnosis labels
    diagnosis_labels = [
        'breast cancer_2019', 'breast cancer_2020', 'breast cancer_2021', 'breast cancer_2022',
        'colon cancer_2019', 'colon cancer_2020','colon cancer_2021', 'colon cancer_2022',
        'lung cancer_2019', 'lung cancer_2020', 'lung cancer_2021', 'lung cancer_2022',]
    # Add 'Diagnosis' column with appropriate labels
    combined_df['Diagnosis'] = diagnosis_labels[:len(combined_df)]
    # Rename the columns for clarity
    combined_df.rename(columns={'Total': 'Treated_patients_TOTAL'}, inplace=True)        # Rename 'Total' to 'Treated_patients_TOTAL'
    combined_df.rename(columns={'All': '30_days_mortality_TOTAL'}, inplace=True)         # Rename 'All' to '30_days_mortality_TOTAL'
    # Reorder the columns to make 'Diagnosis' the first column
    combined_df = combined_df[['Diagnosis', 'Treated_patients_TOTAL', '30_days_mortality_TOTAL', 'Percentage']]
    return combined_df 

def Graph_nb_4_development_30_days_mortality_in_percentage():
    colors = [
        '#ADD8E6',  '#4682B4',  '#1E3A5F',  '#000080',   '#D8A7D3',  '#9B59B6',  '#6A1E9C',  '#4A0072',  '#FFF9C4', '#FFEB3B',  '#FFC107',  '#F57F17',] 
    data = {
        'Diagnosis': ['lung_cancer_2019', 'lung_cancer_2020', 'lung_cancer_2021', 'lung_cancer_2022',
                      'breast_cancer_2019', 'breast_cancer_2020', 'breast_cancer_2021', 'breast_cancer_2022',
                      'colon_cancer_2019', 'colon_cancer_2020', 'colon_cancer_2021', 'colon_cancer_2022'],
        'Percentage': [11.9, 4.2, 2.7, 5.0, 2.3, 0.9, 2.4, 1.0, 0.0, 5.3, 1.2, 2.0]}
    combined_df = pd.DataFrame(data)
    plt.figure(figsize=(12, 6)) 
    plt.gcf().set_facecolor('lightblue')
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # Create the bar plot
    barplot = sns.barplot(data=combined_df, x='Diagnosis', y='Percentage', hue='Diagnosis', palette=colors, errorbar=None)
    # Add the percentages on top of the bars
    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.1f}%',                           # Display percentage with one decimal place
                         (p.get_x() + p.get_width() / 2., p.get_height()),   # Positioning
                         ha='center', va='center',                           # Alignment
                         fontsize=11, color='black',                         # Text settings
                         xytext=(0, 8),                                      # Increase vertical offset (8 points above the bar)
                         textcoords='offset points')
    markers = ['H', 'o', 'D']                                                # 'H' for hexagon, 'o' for circle, 'D' for diamond
    # Split data into 3 groups for different markers
    group1 = combined_df.iloc[:4]     # First 4 diagnoses
    group2 = combined_df.iloc[4:8]    # Next 4 diagnoses
    group3 = combined_df.iloc[8:]     # Last 4 diagnoses
    # Plot lines for each group
    sns.lineplot(data=group1, x='Diagnosis', y='Percentage', marker=markers[0], color='blue', markersize=8, linewidth=2, label='lung_cancer')
    sns.lineplot(data=group2, x='Diagnosis', y='Percentage', marker=markers[1], color='purple', markersize=8, linewidth=2, label='breast_cancer')
    sns.lineplot(data=group3, x='Diagnosis', y='Percentage', marker=markers[2], color='orange', markersize=8, linewidth=2, label='colon_cancer')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Diagnosis name and year')
    plt.ylabel('30 days mortality/treated patients TOTAL %')
    plt.title('Graph nb. 4_Cancer_Development of 30-days mortality in years 2019-2022 in percentage')
    plt.tight_layout()
    plt.show(block=False)

def check_multiple_dates_neutropenia(data_neutropenia_df):
    # Group by 'id' and count unique values in 'date_neutropenia'
    unique_dates_per_id = data_neutropenia_df.groupby('id')['date_neutropenia'].nunique()
    # Filter id that have more than 1 unique date
    ids_with_multiple_dates = unique_dates_per_id[unique_dates_per_id > 1]
    return ids_with_multiple_dates
 
def merge_and_process_date_neutropenia(data_patients_df, data_neutropenia_df):
    # Merge column 'date_neutropenia' from data_neutropenia_df into data_patients_df
    data_patients_df = data_patients_df.merge(data_neutropenia_df[['id', 'date_neutropenia']], on='id', how='left')
    # Convert 'date_neutropenia' to datetime format (day.month.year)
    data_patients_df['date_neutropenia'] = pd.to_datetime(data_patients_df['date_neutropenia'], format='%d.%m.%Y')
    # Move column 'date_neutropenia' to the end of data_patients_df
    datum_column = data_patients_df.pop('date_neutropenia')
    data_patients_df['date_neutropenia'] = datum_column
    return data_patients_df                               

def create_Tab_nb_4_Occurence_of_neutropenia_in_absolute_and_percentage_terms(data_patients_df):
    data_patients_df = data_patients_df[data_patients_df['date_neutropenia'].notna()]                # Remove rows where 'date_neutropenia' is NaN
    data_patients_df['year_neutropenia'] = data_patients_df['date_neutropenia'].dt.year.astype(int)  # Convert 'year_neutropenia' to integer without decimals
    # Create contingency table
    contingency_table_4 = pd.crosstab(
        index=[data_patients_df['diagnosis_name'], data_patients_df['year_neutropenia']],  
        columns='Total',  
        values=data_patients_df['date_neutropenia'], 
        aggfunc='count')
    # create a full range of years (2019, 2020, 2021, 2022) as we can not find year_neutropenia for each year for each diagnosis
    # when we can not find any count, we want to see it like this: 2020, count 0
    full_years = [2019, 2020, 2021, 2022]
    diagnosis_groups = data_patients_df['diagnosis_name'].unique()              # Get unique diagnosis groups
    # create MultiIndex for all combinations of diagnosis groups and years
    full_index = pd.MultiIndex.from_product([diagnosis_groups, full_years], names=['diagnosis_name', 'year_neutropenia'])
    # Reindex contingency table to include all years (even with zero counts)
    contingency_table_4 = contingency_table_4.reindex(full_index, fill_value=0)
    # Ensure `Total` column sums only within each year and diagnosis group, not across years
    contingency_table_4['Total'] = contingency_table_4.groupby(['diagnosis_name', 'year_neutropenia'])['Total'].transform('sum')
    # Reset index to make table more readable
    contingency_table_4 = contingency_table_4.reset_index()
    # Sort by diagnosis and year
    contingency_table_4 = contingency_table_4.sort_values(by=['diagnosis_name', 'year_neutropenia'])
    total_column_values2 = [299, 539, 796, 769, 110, 206, 253, 250, 59, 95, 110, 141]
    # Add this list as the last column in contingency_table_4
    contingency_table_4['Treated_patients_TOTAL'] = total_column_values2
    # Reorder columns 
    cols = [col for col in contingency_table_4.columns if col != 'Treated_patients_TOTAL']
    contingency_table_4 = contingency_table_4[cols + ['Treated_patients_TOTAL']]
    # Rename "Total" column to "Neutropenia_COUNT"
    contingency_table_4 = contingency_table_4.rename(columns={'Total': 'Neutropenia_COUNT'})
    # Calculate percentage and add it as a new column, round 'Percentage' column to 1 decimal place
    contingency_table_4['Occurence of neutropenia_%'] = ((contingency_table_4['Neutropenia_COUNT'] / contingency_table_4['Treated_patients_TOTAL']) * 100).round(1)
    return contingency_table_4

def find_closest_dates(data_patients_df, data_treatment_df):
    data_patients_df['date_neutropenia'] = pd.to_datetime(data_patients_df['date_neutropenia'])
    data_treatment_df['date_of_treatment_application'] = pd.to_datetime(data_treatment_df['date_of_treatment_application'])
    #connect both data frames according to linking element - column'id'
    merged_df = pd.merge(data_patients_df, data_treatment_df, on='id', how='inner')
    merged_df = merged_df.sort_values(by=['id', 'date_of_treatment_application'])
    closest_dates = []
    for id_ in merged_df['id'].unique():
        id_data = merged_df[merged_df['id'] == id_]
        # Ensure we work only with lines, where 'date_neutropenia' is not NaN
        id_data = id_data[pd.notna(id_data['date_neutropenia'])]
        application_dates = data_treatment_df[data_treatment_df['id'] == id_].sort_values(by='date_of_treatment_application')
        application_dates['application_order'] = range(1, len(application_dates) + 1)
        if not id_data.empty:
            row = id_data.iloc[0]
            previous_application = application_dates[application_dates['date_of_treatment_application'] < row['date_neutropenia']]
            if not previous_application.empty:
                closest_row = previous_application.loc[(row['date_neutropenia'] - previous_application['date_of_treatment_application']).idxmin()]
                closest_dates.append({
                    'id': row['id'],
                    'date_neutropenia': row['date_neutropenia'],
                    'date_previous_application': closest_row['date_of_treatment_application'],
                    'application_order': closest_row['application_order'],  
                })
            else:
                closest_dates.append({
                    'id': row['id'],
                    'date_neutropenia': row['date_neutropenia'],
                    'date_previous_application': pd.NaT, 
                    'application_order': None,  
                })
    closest_dates_df = pd.DataFrame(closest_dates)
    return closest_dates_df

def find_most_frequent_values(final_table):
    # Find 5 the most common values in 'application_order' and their counts
    most_common_application_order = final_table['application_order'].value_counts().nlargest(5)
    # Create dictionary for conversion to table
    result_dict = {
        'order of therapy application': most_common_application_order.index,
        'number of occurences': most_common_application_order.values
    }
    result_df = pd.DataFrame(result_dict)                 # Convert dictionary to DataFrame
    # values to be integers
    result_df['number of occurences'] = result_df['number of occurences'].astype('Int64')
    result_df['order of therapy application'] = result_df['order of therapy application'].astype('Int64')
    return result_df

def Graph_nb_5_Incidence_of_Febrile_Neutropenia_by_Application_Order(result_df):
    plt.figure(figsize=(8, 6))                        # Create a bar plot for number of occurrences
    plt.bar(result_df['order of therapy application'], result_df['number of occurences'], color='darkblue')
    # Add a trend line with orange color and markers
    plt.plot(result_df['order of therapy application'], result_df['number of occurences'], color='orange', marker='o', label='Trend Line')
    plt.gca().set_facecolor('lightgray')              # Set background color for the plot
    plt.xlabel('Order of Therapy Application')
    plt.ylabel('Number of Occurrences')
    plt.title('Graph nb. 5_Incidence of febrile neutropenia by order of treatment application')
    plt.xticks(result_df['order of therapy application'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

# Run the script
if __name__ == "__main__":
    main()                                        # Calls the main function when the script is executed