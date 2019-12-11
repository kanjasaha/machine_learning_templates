##Data Exploration/Clean Up/Transformation
#data manipulation libraries
import pandas as pd
import numpy as np
import itertools as iter
from scipy import stats

#Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from IPython.display import display,Markdown, HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn.decomposition import PCA
import plotly.graph_objs as go
from scipy.stats import chi2_contingency
import matplotlib as mpl
mpl.rc("figure", figsize=(20, 10))

class explore(object):
    def __init__(self, df,index_variable=None,label=None,include_variables=None,exclude_variables=None,date_variable=None):
        self.dataframe=df.copy()
        if index_variable is None:
            self.index_variable=self.dataframe.columns[0]
        else:
            self.index_variable=index_variable
        self.dataframe.drop(columns=self.index_variable, inplace=True)
        self.label=label
        
        self.date_variable=date_variable
        
        if (include_variables is not None):    
            self.include_variables=include_variables
        else:
            self.include_variables=self.dataframe.columns
        self.exclude_variables=exclude_variables
        self.categorical_variables=self.get_categorical_variables()
        self.numerical_variables=self.get_numerical_variables()
       
    
    

    def get_categorical_variables(self): 
        cat_variables=list(self.dataframe.select_dtypes(include=[np.object,np.bool]).columns)
        if self.index_variable in cat_variables:
            cat_variables.remove(self.index_variable)
        
        if self.label is not None:
            if self.label in cat_variables :
                cat_variables.remove(self.label)

        if self.exclude_variables is None:
            cat_variables_filtered=cat_variables
        else:
            cat_variables_filtered=[elem for elem in cat_variables if elem not in self.exclude_variables] 
        return cat_variables_filtered;

    def get_numerical_variables(self): 
        numerical_variables=list(self.dataframe.select_dtypes(include=[np.float,np.int]).columns)
        if self.index_variable in numerical_variables:
            numerical_variables.remove(self.index_variable)
        
        if self.label is not None:
            if self.label in numerical_variables :
                numerical_variables.remove(self.label)

        if self.exclude_variables is None:
            numerical_variables_filtered=numerical_variables
        else:
            numerical_variables_filtered=[elem for elem in numerical_variables if elem not in self.exclude_variables] 
        return numerical_variables_filtered;

    def data_sample(self):    
        # a quick look at the sample of the data set
        display (" dataset has {} rows(samples) with {} columns(features) each.".format(*self.dataframe.shape))
        display(('--------------------Sample Dataset--------------------'))
        display(self.dataframe.head().transpose())

    def summary_plots(self, **kwargs):

        display(('--------------------Boxplot of Numeric Features--------------------'))
        plt.show(self.dataframe.boxplot(rot=45))
        display(('--------------------Mean and standard deviation Numeric Features--------------------'))
        #dd=train.iloc[:,100:200].describe().transpose().reset_index()
        dd=self.dataframe.describe().transpose().reset_index()
        plt.errorbar(range(len(dd['index'])), dd['mean'],markersize=10, capsize=4,yerr=dd['std'], fmt='o')
        plt.xticks(range(len(dd['index'])), dd['index'],rotation=45)
        plt.show()

        #display(dd) 

    

    def summary_table(self): 
        df_stats=self.dataframe.describe(include="all", percentiles=(0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1)).transpose() 
        df_null_count=self.dataframe.isnull().sum() 
        df_null_count_percent=(df_null_count*100.0)/self.dataframe.shape[0] 
        df_0_count=self.dataframe.eq(0).sum() 
        df_0_count_percent=(df_0_count*100.0)/self.dataframe.shape[0] 
        df_nan_count=self.dataframe.isin([np.nan, np.inf, -np.inf]).sum()
        df_nan_count_percent=(df_nan_count*100.0)/self.dataframe.shape[0] 
        df_final=pd.concat([df_null_count,df_null_count_percent,df_0_count,df_0_count_percent,df_nan_count,df_nan_count_percent,df_stats],axis=1,join='outer') 
        df_final.rename(columns={0:'Null Values',1:'% of null values',2:'0 Values',3:'% of 0 Values',4:'NaN Values',5:'% of NaN Values'}, inplace=True) 
        display(df_final) 

    def unique_value_count(self):   
        #Information on the data type
        display(('--------------------Plot Unique Value Counts and DataType for each Feature--------------------'))
        col_info=pd.DataFrame(columns=['col_name','label','unique_values'])
        for col in self.dataframe.columns:
            row = [col, self.dataframe[col].dtype , self.dataframe[col].nunique()]
            col_info.loc[len(col_info)] = row

        col_info['label']=col_info['label'].astype('str')
        sns.catplot(x='col_name', y='unique_values', hue='label', kind='bar', data=col_info, height=6, aspect=col_info.shape[0]/10+2)
            
        plt.xticks(rotation=90)
        plt.show()

        display(('--------------------Unique Values for each categorical feature--------------------'))
        for col in list(self.dataframe.select_dtypes(include=[np.object,np.bool]).columns):
            print('Column Name: {} : {}'.format(col,self.dataframe[col].unique()))
        #display(('--------------------Unique Value Counts and DataType for each Feature--------------------'))
        #for col in list(self.dataframe.columns):
        #    print('Column Name: {} DataType: {} : {}'.format(col,self.dataframe[col].dtype,self.dataframe[col].nunique()))
    
    def missing_value_count(self): 
        display(('--------------------% of 0 Values per column--------------------'))
        num_col_0_values=self.dataframe.columns[self.dataframe.eq(0).any()].size
        print('Number of columns with 0 values : {}'.format(num_col_0_values))
        col_info=pd.DataFrame(columns=['col_name','zero_value_pct'])
        for col in self.dataframe.columns:
              row = [col, (self.dataframe[col].eq(0).sum()*100.0)/self.dataframe.shape[0]]
              col_info.loc[len(col_info)] = row

        sns.catplot(x='col_name', y='zero_value_pct', kind='bar', data=col_info, height=6, aspect=round(num_col_0_values/10)+2)
        plt.xticks(rotation=90)
        plt.show()

        #for col in self.dataframe.columns[self.dataframe.eq(0).any()].tolist():
        #    print('{} : {} % '.format(col,(self.dataframe[col].eq(0).sum()*100.0)/self.dataframe.shape[0]))

        display(('--------------------% of Null Values per column--------------------'))
        num_col_null_values=self.dataframe.columns[self.dataframe.isnull().any()].size
        
        print('Number of columns with null values : {}'.format(num_col_null_values))
        col_info=pd.DataFrame(columns=['col_name','null_value_pct'])
        for col in self.dataframe.columns:
              row = [col, (self.dataframe[col].isnull().sum()*100.0)/self.dataframe.shape[0]]
              col_info.loc[len(col_info)] = row

        #col_info['label']=col_info['label'].astype('str')
        sns.catplot(x='col_name', y='null_value_pct', kind='bar', data=col_info, height=6, aspect=round(num_col_null_values/10)+2)
        plt.xticks(rotation=90)
        plt.show()
        #for col in self.dataframe.columns[self.dataframe.isnull().any()].tolist():
        #    print('{} : {} % '.format(col,(self.dataframe[col].isnull().sum()*100.0)/self.dataframe.shape[0]))

        display(('--------------------% of NaN/Inf Values per column--------------------'))
        num_sql_nan_inf_values=self.dataframe.isin([np.nan, np.inf, -np.inf]).any().size
        print('Number of columns with Nan/Inf Values :{}'.format(num_sql_nan_inf_values))
        
        col_info=pd.DataFrame(columns=['col_name','nan_inf_value_pct'])
        for col in self.dataframe.columns:
              row = [col, (self.dataframe[col].isin([np.nan, np.inf, -np.inf]).sum()*100.0)/self.dataframe.shape[0]]
              col_info.loc[len(col_info)] = row

        sns.catplot(x='col_name', y='nan_inf_value_pct', kind='bar', data=col_info, height=6, aspect=round(num_sql_nan_inf_values/10)+2)
        plt.xticks(rotation=90)
        plt.show()

        #for col in self.dataframe.columns[self.dataframe.isin([np.nan, np.inf, -np.inf]).any()].tolist():
        #    print('{} : {} % '.format(col,(self.dataframe[col].isin([np.nan, np.inf, -np.inf]).sum()*100.0)/self.dataframe.shape[0]))
            
    def duplicate_value_count(self):              
        display(('--------------------Duplicate Rows--------------------'))
        if (self.dataframe.duplicated().any() == True):
          display("There are {} duplicate rows".format(self.dataframe[self.dataframe.duplicated()].shape[0]))
        else:
          display("There are no duplicates")

      
    
    def show_numerical_variable_plots(self):
        self.plot_hist()
        self.box_plot()
        self.scatter_plot_independent_label()
        sns.set()
        sns.heatmap(self.dataframe.corr(),annot=True,cmap="Blues",fmt=".2f",linewidths=.05)
        self.pearson_correlation_matrix()
        self.check_for_outliers();
        
    def show_categorical_variable_plots(self):
        self.percent_bar_chart()
        self.chi_square_test()
        self.bar_chart()
        self.cat_plot_independent_label() 
        
    
        
    def plot_hist(self):
        display(Markdown('**--------------------Histograms of each Feature--------------------**'))
       
        for col in self.numerical_variables:
            
            s=self.dataframe[col].dropna()
            #w = 100*(np.zeros_like(s) + 1. / len(s))
            plt.hist(s)
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.title('Histogram of '+ col)
            plt.grid(True)
            plt.show()
    
    def sns_heatmap(self):
        sns.set()
        sns.heatmap(self.dataframe.corr(),annot=True,cmap="Blues",fmt=".2f",linewidths=.05)
        plt.show()
        
    def bar_chart(self):
        display(('--------------------bar chart of categorial variables--------------------'))
   
        for x in self.categorical_variables:
           # if x != y:
            sns.catplot(x=x, kind="count", palette="deep", data=self.dataframe,height=5, aspect=2);
            plt.xlabel(x)
            plt.xticks(rotation='vertical')
            plt.ylabel('Frequency')
            plt.title('Bar chart of '+ x ,size=16, y=1.05)
            plt.grid(True)
            plt.show()
            display('==============================================================================================================')

    def plot_density(self):
        display(Markdown('**--------------------Density Plot of each Feature--------------------**'))
   
        for col in self.numerical_variables:
            plt.hist(self.dataframe[col], bins=10)
            plt.xlabel(col)
            plt.ylabel('Frequency')
            sns.distplot(self.dataframe[col], hist=True, kde=True, 
                 bins=int(180/5), color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 4})
            plt.grid(True)
            plt.show()
            
    def pair_plot(self):
        display(Markdown('**--------------------Pairplot of all Features with continous numeric value--------------------**'))
        for x,y in iter.combinations(self.include_variables,2):
            #if x != y:
                plt.scatter(self.dataframe[x],self.dataframe[y])
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title('Pairplot of '+ x + ' and ' + y)
                plt.grid(True)
                plt.show()

    '''def bar_chart_xyz(self):
        display(Markdown('**--------------------bar plot of each column segmented by another column--------------------**'))

        for x,y,z in iter.combinations(self.include_variables,3):
           # if x != y:
            sns.catplot(x=x, hue=y,col=z, kind="count", palette="deep", data=self.dataframe,height=20, aspect=2);
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Bar plot of '+ x + ' and ' + y)
            plt.grid(True)
            plt.show()'''

    def retention_funnel(self,funnel_level,aggregate_column):
        display(Markdown('**--------------------retention funnel of each column by billing cycles--------------------**'))

        for x in self.include_variables:
            if x !=funnel_level and x!=aggregate_column:
                df_group=self.dataframe.groupby([funnel_level,x])[aggregate_column].agg('count').unstack()
                pct=round((df_group.pct_change()+1)*100,0)
                #display(pct)
                pct.plot()
                plt.xlabel(x)     

    def bar_chart_2_variables(self):
        display(('--------------------bar plot of each categorical variables segmented by another categorical variables--------------------'))

        for x,y in iter.combinations(self.include_variables,2):
           # if x != y:
            sns.catplot(x=x, hue=y, kind="count", palette="deep", data=self.dataframe,height=5, aspect=2);
            plt.xticks(rotation='vertical')
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Bar plot of '+ x + ' and ' + y,size=16, y=1.05)
            plt.grid(True)
            plt.show()

    def bar_chart_independent_label(self):
        display(('--------------------bar plot of each categorical variables segmented by label--------------------'))
        y=label;
        for x in self.include_variables:
           # if x != y:
            sns.catplot(x=x, hue=y, kind="count", palette="deep", data=self.dataframe,height=5, aspect=2);
            plt.xticks(rotation='vertical')
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Bar plot of '+ x + ' and ' + y,size=16, y=1.05)
            plt.grid(True)
            plt.show()
    
    

            
    def bar_chart(self):
        display(Markdown('**--------------------bar plot of each column segmented by another column--------------------**'))

        for x,y in iter.combinations(self.include_variables,2):
           # if x != y:
            sns.catplot(x=x, hue=y, kind="count", palette="deep", data=self.dataframe,height=5, aspect=2);
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Bar plot of '+ x + ' and ' + y)
            plt.grid(True)
            plt.show()

    def percent_bar_chart(self):
        display(Markdown('**--------------------bar plot of each column segmented by another column--------------------**'))

        for x,y in iter.combinations(self.include_variables,2):
           # if x != y:
            freq_df = self.dataframe.groupby([x])[y].value_counts(normalize=True).unstack()
            pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)
            pct_df.plot(kind="bar")
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.0))
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Bar plot of '+ x + ' and ' + y)
            plt.grid(True)
            plt.show()

    def cat_plot_independent_label(self):
        display(('--------------------bar plot of each categorical variables segmented by another categorical variables--------------------'))
        y=self.label;
        for x in self.categorical_variables:
           # if x != y:
            #sns.catplot(x=x, hue=y, kind="count", palette="deep", data=self.dataframe,height=5, aspect=2);
            sns.catplot(x=x, y=y, data=self.dataframe,height=5, aspect=2);
            plt.xticks(rotation='vertical')
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Bar plot of '+ x + ' and ' + y,size=16, y=1.05)
            plt.grid(True)
            plt.show()
    
    def reg_line_residual_independent_label(self):
        display(('--------------------reg line numerical variables by label--------------------'))
        x=self.label;
        for y in self.numerical_variables:
            fig, (ax1, ax2) = plt.subplots(2)
            sns.regplot(x=x, y=y,data=self.dataframe, color="b",ax=ax1);
            sns.residplot(x, y, data=self.dataframe,ax=ax2)
            plt.xticks(rotation='vertical')
           # plt.xlabel(x, y=.5)
            plt.ylabel(y)
            ax1.set_title('regression line', y=.9)
            ax2.set_title('residual plot', y=.9)
            #fig.suptitle('Scatter plot of '+ x + ' and ' + y,size=16, y=1.05)
            plt.grid(True)
            plt.show()
      
    def r2(x, y):
        return stats.pearsonr(x, y)[0] ** 2
            
   
    
    def scatter_plot_independent_label(self):
        display(('--------------------scatter plot numerical variables by label--------------------'))
        x=self.label;
        for y in self.numerical_variables:
           # if x != y:
            
            sns.scatterplot(x=x, y=y, palette="deep", data=self.dataframe);
            plt.xticks(rotation='vertical')
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Scatter plot of '+ x + ' and ' + y,size=16, y=1.05)
            plt.grid(True)
            plt.show()
            
    

    def percent_bar_chart(self):
        display(('--------------------bar plot of each column segmented by another column--------------------'))

        for x,y in iter.combinations(self.include_variables,2):
           # if x != y:
            freq_df = self.dataframe.groupby([x])[y].value_counts(normalize=True).unstack()
            pct_df = freq_df.divide(freq_df.sum(axis=1), axis=0)
            pct_df.plot(kind="bar")
            plt.gca().yaxis.set_major_formatter(IndexFormatter(1))
            plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.0))
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title('Bar plot of '+ x + ' and ' + y,size=16, y=1.05)
            plt.grid(True)
            plt.show()       

    def percentage_barh_stacked_chart(self):
        display(Markdown('**--------------------percentage chart for categorical variables--------------------**'))

        for x,y in iter.combinations(self.include_variables,2):
            #if x != y:
                cont_table=pd.crosstab(self.dataframe[x],self.dataframe[y], normalize='index')
                #print(cont_table)
                cont_table.plot(kind='barh', stacked=True)
                plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.0))
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title('Bar plot of '+ x + ' and ' + y)
                plt.grid(True)
                plt.show()                


    def time_series_feature_label(self):
        for n in self.numerical_variables:
          plt.xlabel('Time series of target and independent variables')
          df=self.dataframe
          df[self.date_variable]=pd.to_datetime(df[self.date_variable])
          df.index=df[self.date_variable]
          feature_plot=df[n]
          target_plot=df[self.label]
          ax1 = feature_plot.plot(color='blue',x=self.date_variable, grid=True, label=n)
          ax2 = target_plot.plot(color='red', x=self.date_variable, grid=True, secondary_y=True, label=self.label)
          h1, l1 = ax1.get_legend_handles_labels()
          h2, l2 = ax2.get_legend_handles_labels()
          plt.legend(h1+h2, l1+l2, loc=2)
          plt.show()
    
    def seasonality(self):
      display(('--------------------Seasonality plot of each variable --------------------'))
      for n  in self.numerical_variables:
              season = self.dataframe.copy()
              season['Year'] = season['Date'].dt.year
              season['Day'] = season['Date'].dt.dayofyear
              spivot = pd.pivot_table(season, index='Day', columns = 'Year', values = n)
              spivot.plot(linewidth=3)
              plt.title('Seasonality plot of '+ n + ' and ' + self.date_variable,size=16, y=1.05)
              plt.show()
              
    def time_series_numerical_categorical(self):
        display(('--------------------bar plot of each numerical variables segmented by a categorical variables--------------------'))
        for a,b in [(m,n) for m in self.categorical_variables   for n in self.numerical_variables]:
           # if x != y:
            
            sns.lineplot(data=self.dataframe, x=self.date_variable, hue=a,  y=b)
           
            plt.title('Time series plot of '+ b + ' and ' + self.date_variable,size=16, y=1.05)
            plt.ylabel(b)
            plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.0))
            plt.xlabel(self.date_variable)
            plt.xticks(rotation=90)
            plt.grid(True)
            plt.show()
            
            
    

    def time_series(self):
        display(('--------------------time series plot of each variable --------------------'))

        for n in self.numerical_variables:
                #plt.figure(figsize=(15,10))
                self.dataframe.plot(x=self.date_variable, y=n)
                plt.title('Time series plot of '+ n + ' and ' + self.date_variable,size=16, y=1.05)
                plt.ylabel(n)
                plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.0))
                plt.xlabel(self.date_variable)
                plt.xticks(rotation=90)
                plt.grid(True)
                plt.show()

    def line_plot_multiple(self):
        display(('--------------------time series plot of each variable --------------------'))

        for x in self.categorical_variables:
                
                df_d=self.dataframe.groupby([timeseries,x], as_index=False)[value].agg({'group_size':'sum'})
                sns.lineplot(data=df_d, x=timeseries, hue=x,  y="group_size")
                plt.title('Time series plot of '+ x + ' and ' + timeseries,size=16, y=1.05)
                plt.ylabel(x)
                plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.0))
                plt.xlabel(timeseries)
                plt.grid(True)
                plt.show()


    def line_plot(self):
        display(Markdown('**--------------------box plot of each column segmented by another column--------------------**'))

        for x in self.categorical_variables:
                
                df_d=self.dataframe.groupby([timeseries,x], as_index=False)[value].agg({'group_size':'sum'})
                sns.lineplot(data=df_d, x=timeseries, hue=x,  y="group_size") 
                plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.0))
                plt.ylabel(x)
                plt.xlabel(timeseries)
                plt.title('Time series plot of '+ x + ' and ' + timeseries)
                plt.grid(True)
                plt.show()

    def box_plot(self):
        display(Markdown('**--------------------box plot of each column segmented by another column--------------------**'))

        for x, y in ([(m , n) for m in self.categorical_variables for n in self.numerical_variables ]):
                
                sns.boxplot(x=x, y=y, linewidth=2.5,data=self.dataframe)
                plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.0))
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title('Box plot of '+ x + ' and ' + y)
                plt.grid(True)
                plt.show()

    def box_plot_categorical_label(self):
        display(('--------------------box plot of dependent varaible for categorical variable--------------------'))

        for y in self.categorical_variables:
                #sns.catplot(y=label, x=y, linewidth=2.5, kind="box",data=self.dataframe,height=5, aspect=2)
                plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.0))
                plt.xticks(rotation='vertical')
                plt.xlabel(label)
                plt.ylabel(y)
                sns.set_style("whitegrid")
                ax = sns.boxplot(x=y, y=label, data=self.dataframe)

                plt.title('Box plot of '+ y,size=16, y=1.05 )
                medians = self.dataframe.groupby([y])[label].median().values
                median_labels = [str(np.round(s, 2)) for s in medians]

                pos = range(len(medians))
                for tick,label in zip(pos,ax.get_xticklabels()):
                    ax.text(pos[tick], medians[tick] + 0.5, median_labels[tick], 
                    horizontalalignment='center', size='x-small', color='w', weight='semibold')
                plt.grid(True)
                plt.show()

    def pearson_correlation_matrix(self):
        display(('--------------------pearson correlation--------------------'))
        corr_df = pd.DataFrame(columns=['feature', 'correlation', 'p_value'])
        rows_list = []
        y=self.label
        for x in self.numerical_variables:
         # if x != y:
          display('Pearson Correlation between ' + x + ' and ' + y);
          display(pearsonr(self.dataframe[x],self.dataframe[y]));
          display('-----------------------------------------------');
          correlation,p_value=pearsonr(self.dataframe[x],self.dataframe[y])
          dict1 = {}
          dict1 = {'column':x,'correlation':correlation,'p_value':p_value}
          rows_list.append(dict1)

        corr_df = pd.DataFrame(rows_list);  
        display(corr_df);

    def correlation_matrix_continuous(self):
        display(Markdown('**--------------------Correlation matrix continous numeric features--------------------**'))

        f, ax = plt.subplots(figsize=(10, 6))
        hm = sns.heatmap(round(df.corr(),2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                     linewidths=.05)
        f.subplots_adjust(top=0.93)
        t= f.suptitle('Correlation Heatmap', fontsize=14)  
  
    def chi_square_test(self):
        display(Markdown('**--------------------percentage chart for categorical variables--------------------**'))
        #final_columns=[col for col in include_columns if col not in exclude_columns]

        for x,y in iter.combinations(self.include_variables,2):
            #if x != y:
                crosstable=pd.crosstab(self.dataframe[x],self.dataframe[y], normalize='index')
                cont_table=pd.crosstab(self.dataframe[x],self.dataframe[y])
                #print(cont_table)
                crosstable.plot(kind='barh', stacked=True,figsize=(10,5))
                plt.legend(loc="upper right", bbox_to_anchor=(1.2,1.0))
                #plt.xlabel(x)
                plt.ylabel(x)
                plt.title('Bar plot of '+ x + ' and ' + y)
                plt.grid(True)
                plt.show()
                chi2,p,df=chi2_contingency(cont_table)[0:3]
                print('**The Null and Alternate Hypotheses**')

                print('H0:There is no statistically significant relationship between the two selected variables')
                print('Ha:There is a statistically significant relationship between the two selected variables')



                if p < .05:
                    print("We can reject the Null Hypothesis and say that " + x + " and " + y + " have some relationship")
                else:
                    print("We cannot reject the Null hypothesis and say that " + x + " and " + y + " are truly independent")
                print()
                print ("X2: {}, p-value: {}, Degrees of Freedom: {}".format(chi2,p,df))
    
    def outliers(self,drop_outlier=False):
    
        # For each feature find the data points with extreme high or low values
        log_data=self.dataframe
        x=[]
        for feature in log_data.keys():
    
            # TODO: Calculate Q1 (25th percentile of the data) for the given feature
            Q1 = np.percentile(log_data[feature],25)

            # TODO: Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile(log_data[feature],75)

            # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
            step = 1.5*(Q3-Q1)
            y= log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
            y1=y.index.values
            x.append(y1)
            # Display the outliers
            #outliercount=y.shape[0]
            #print ("'{} Data points considered outliers for the feature '{}':".format(outliercount,feature))


            # OPTIONAL: Select the indices for data points you wish to remove
            # Here I go through the lists and extract the index value that is repeated in more than one list.
            seen = set()
            repeated = set()
        for l in x:
            for i in set(l):
                if i in seen:
                  repeated.add(i)
                else:
                  seen.add(i)

        outliers =list(repeated)
        outlier_count=len(outliers)
        total_count=len(log_data)
       
        percent_outliers=(float(outlier_count)*100)/(float(total_count))
        #display(percent_outliers)
        delete_status = "Outlier not dropped from dataset"

        if drop_outlier is True:
            # Remove the outliers, if any were specified
            good_data = data.loc[~data.index.isin(outliers)]
            delete_status = "Outlier Dropped from dataset"
            data=good_data

        message =("{} ({:2.2f}%) data points considered outliers from the dataset of {}. {}.".format(outlier_count,percent_outliers,total_count,delete_status))   
        return data,outliers , message


    def pca_results(self,n_components=6):

        pca = PCA(copy=True, iterated_power='auto', n_components=n_components, random_state=42,
        svd_solver='auto', tol=0.0, whiten=False)
        df_pca=self.dataframe[self.numerical_variables]
        pca.fit(df_pca)
      

        dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

        # PCA components
        components = pd.DataFrame(np.round(pca.components_, 4), columns = list(df_pca.keys()))
        components.index = dimensions

        # PCA explained variance
        ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
        variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
        variance_ratios.index = dimensions

        # Create a bar plot visualization
        fig, ax = plt.subplots(figsize = (14,8))

        # Plot the feature weights as a function of the components
        components.plot(ax = ax, kind = 'bar');
        ax.set_ylabel("Feature Weights")
        ax.set_xticklabels(dimensions, rotation=0)


          # Display the explained variance ratios
        for i, ev in enumerate(pca.explained_variance_ratio_):
            ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

        # Return a concatenated DataFrame
        pca_results=pd.concat([variance_ratios, components], axis = 1)
        #pca_results_cumsum=pca_results['Explained Variance'].cumsum()
        return pca

    