import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import statsmodels.api as sm
from itertools import combinations

# The Critical Values of the Durbin Watson Test are from:
# https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/regression/supporting-topics/model-assumptions/test-for-autocorrelation-by-using-the-durbin-watson-statistic/
durbin_watson = pd.read_csv('/Users/macbook/Desktop/Proyecto Métodos Estadísticos/Regression-Modeling/critical_values_durbin_watson.csv')

# THIS FUNCTIONS ARE GOING TO BE GLOBAL
class global_functions:
    # Function for selecting the "y" column in the data frame (dependient variable)
    def select_y_var(df):
        [print(f"{i+1}. {k}") for i,k in enumerate(df.columns)]
        x = input("Select y variable:")
        y_index = int(x)-1
        y_name = df.iloc[:,y_index:y_index+1].columns[0]
        return {'y_index':int(x)-1,'y_name':y_name}

    # Function to make the different scatter plots MULTIPLE AND SIMPLE REGRESSION MODELS
    def scatter_plot(df,title):
        # First Column of the data frame is going to be X axis an second column is going to be Y axis
        x_name = df.columns[0]
        y_name = df.columns[1]

        # Creating Scatter Plot
        fig = px.scatter(df, x=x_name, y=y_name, template='plotly_dark', 
                        color=y_name, hover_data={x_name: True, y_name: True})


        # Configurar el título con Plotly Graph Objects
        fig.update_layout(title=dict(text=f"<b>{title}</b>",
                                    x=0.5,  # Centering title horizontaly
                                    y=0.95,  # Title aligned to the top of the plot
                                    xanchor='center',  # Horizontal anchor cenetered
                                    yanchor='top'))  # Vertical anchor at the top

        # Updating tooltip, color and size of the dots
        fig.update_traces(marker=dict(color='blue', size=8), hovertemplate=f'{x_name}: %{{x}}<br>{y_name}: %{{y}}', line=dict(dash='dot'))
        
        # Just for the Versus Fits and the Versus Order Plots, we are adding an horizontal dotted line.
        if title == 'Versus Fits' or title == 'Versus Order':
            # Adding horizontal dotted line.
            fig.add_shape(type='line',
                        x0=df[x_name].min(), x1=df[x_name].max(),
                        y0=0, y1=0,  # Adjusting de horizontal line to y=0
                        line=dict(color='white', dash='dot'))
        
        # Just for the Versus Order Plot, we are adding a line that follows all the points without smoothing lines
        if title == 'Versus Order':
            line_fig = px.line(df, x=x_name, y=y_name)
            for trace in line_fig.data:
                trace.update(line=dict(color='blue', width=2))
                fig.add_trace(trace)
        
        # Just for the Normal Probability Plot we are adding a tendence line.
        if title == 'Normal Probability Plot' or title == 'Correlation Plot':      
            # Calcular la línea de tendencia usando statsmodels
            X = sm.add_constant(df[x_name])  # Agregar una constante para el término independiente
            model = sm.OLS(df[y_name], X).fit()
            trendline = model.predict(X)
            if title == 'Normal Probability Plot':
                # Agregar la línea de tendencia al gráfico
                fig.add_trace(go.Scatter(x=df[x_name], y=trendline, mode='lines', name='OLS', line=dict(color='red', dash='dot')))
                fig.update_layout(showlegend=False)
            else:
                # Agregar la línea de tendencia al gráfico
                fig.add_trace(go.Scatter(x=df[x_name], y=trendline, mode='lines', name='OLS', line=dict(color='white', dash='dot')))
                fig.update_traces()
                fig.update_layout(showlegend=False, title=dict(text=f"<b>Correlation Plot ({df.corr()[df.columns[0]][df.columns[1]]*100:.2f}%)</b>", x=.5))
        return fig

    # RESIDUALS DISTRIBUTED NORMAL WITH MEAN 0, VARIANCE OF THE RESIDUALS
    # NORMAL DISTRIBUTION MEAN 0
    # H0: Residuals come from a Normal distribution with 0 mean
    # Ha: Residuals come from anothe distribution
    def normality_0_mean(anova_table,residuals):
        n = len(residuals)
        classes = int(n**(1/2))

        # Funcion Para Redondear hacia Arriba
        def redondear_hacia_arriba(numero, decimales=4):
            if isinstance(numero, float):
                parte_decimal = numero - int(numero)
                if parte_decimal > 0:
                    redondeado = math.ceil(numero * 10**decimales) / 10**decimales
                    return round(redondeado, decimales)
            return numero

        width = redondear_hacia_arriba((residuals.max() - residuals.min())/classes)
        linf = [residuals.min()] + [residuals.min() + width*i for i in range(1,classes)]
        lsup = [residuals.min() + width*i for i in range(1,classes)] + [residuals.max()]
        
        limits = [residuals.min()] + [residuals.min() + width*i for i in range(1,classes)] + [residuals.max()]
        intervals = [f"({limits[i].__round__(4)}, {limits[i+1].__round__(4)})" for i in range(len(limits)-1)]
        
        # List of Frequencies by Interval
        frequence = [0]*classes

        # Counting Frequence in each Interval
        for value in residuals:
            for i in range(classes):
                if linf[i] <= value < lsup[i]:
                    frequence[i] += 1
                    break
        frequence[-1] += 1
        
        # Data frame with frecuence and interval of values, for histogram
        histogram_df = pd.DataFrame({'Residuals':intervals,'Frequence':frequence})
        
        # Histogram. Using function in class plots
        histogram_plt = plots.histogram(histogram_df, 'Histogram')
        
        # Residuals Standard Deviation
        residuals_std = (anova_table['half_square'][1])**(1/2)

        # Cumulative Distribution Function
        cdf = [stats.norm.cdf(lsup[0], scale=residuals_std, loc=0)] + [stats.norm.cdf(lsup[i], scale=residuals_std, loc=0) - stats.norm.cdf(linf[i], scale=residuals_std, loc=0) for i in range(1,classes-1)] + [1-stats.norm.cdf(lsup[classes-2], scale=residuals_std, loc=0)]

        # Expected Frequency
        exp_freq = [i*n for i in cdf]

        quotient = [np.square(exp_freq[i] - frequence[i])/exp_freq[i] for i in range(classes)]

        test_statistic = sum(quotient)

        # Chi Square Inverse Functions with alpha = 0.05 and n = classes - 2
        chi_square = stats.chi2.isf(.05, classes-2)

        if test_statistic > chi_square:
            normal_0_mean_hypothesis = f"{test_statistic.__round__(4)} > {chi_square.__round__(4)}, Residuals come from another distribution"
        else:
            normal_0_mean_hypothesis = f"{test_statistic.__round__(4)} < {chi_square.__round__(4)}, Residuals come from a Normal distribution with mean 0"
            
        return normal_0_mean_hypothesis, histogram_plt
    
    def incorrelation_assumption(residuals,k):
        # INCORRELATION ASSUMPTION
        # H0: p = 0, which means the analysed data doesn't have correlation
        # Ha: p > 0, which means the analysed data has correlation
        # Reject H0 if d < dL
        # Don't rehect H0 if d > dU
        # Inconclusive test if dL < d < dU
        n = len(residuals)
        
        d = sum(np.square([residuals[i+1] - residuals[i] for i in range(n-1)]))/sum(np.square(residuals))

        finding_row = durbin_watson.loc[(durbin_watson['sample_size'] == n) & (durbin_watson['n_terms'] == k)].reset_index(drop=True)
        dL = finding_row['DL'][0]
        dU = finding_row['DU'][0]

        if d < dL:
            incorrelation_test = f"{d} < {dL}, Correlated data."
        elif d > dU:
            incorrelation_test = f"{d} > {dU}, Incorrelated data."
        else:
            incorrelation_test = f"{dL} < {d} < {dU}, Inconclusive test."
            
        return incorrelation_test

    # ATYPICAL DATA IN THE SAMPLE
    def atypical_data(residuals, residuals_var, hi):
        # ATYPICAL DATA
        # Standarized Residuals
        standarized_residuals = [(value)/(residuals_var*(1-hi[i]))**.5 for i,value in enumerate(residuals)]
        # Finding Atypical Data
        atypical_data = {'Obs':[],'Resid':[],'Std Resid':[],'hi':[]}
        for i,value in enumerate(standarized_residuals):
            if abs(value) > 3:
                atypical_data['Obs'].append(i+1)
                atypical_data['Resid'].append(residuals[i])
                atypical_data['Std Resid'].append(value)
                atypical_data['hi'].append(hi[i])
        atypical_data = pd.DataFrame(atypical_data)
        
        # If there is not atypical data, it will tell the user
        if len(atypical_data['Obs']) == 0:
            atypical_data = "There is not atypical data in the sample"
            
        return atypical_data
    
    def qqplot(residuals,residuals_variance):
        # NORMALITY OF THE RESIDUALS
        # Sorting residuals ascendently
        sorted_residuals = np.sort(residuals)
        # k is the probability P[Z < Zk] = k
        k = [(i-.375)/(len(residuals)+.25) for i in range(1,len(residuals)+1)]
        # Now we find Zk
        Zk = [stats.norm.isf(1-i) for i in k]
        # Finally we calculate the expected value if the residuals where distributed normally
        exp_value = [i*residuals_variance**.5 for i in Zk]
        # Data Frame to use the scatter plot function
        qqplot_df = pd.DataFrame({'Residuals':sorted_residuals,'Expected Value':exp_value})

        # Normal Probability Plot
        normal_prob_plt = global_functions.scatter_plot(qqplot_df,'Normal Probability Plot')
        
        return normal_prob_plt

# THIS FUNCTIONS CAN BE USED IN THE SIMPLE REGRESSION MODELING, SOME OF THEM IN THE MULTIPLE REGRESSION MODELING
class linearization:
    # Function to determine the coefficients and the estimated y (JUST FOR SIMPLE REGRESSION MODELING)
    def linearizable_model(df):
        # LINEAR MODEL: y = b0 + b1 * x
        # ln(y) = ln(b0) + ln(b1) + ln(x)
        # Values of the constants of the linear model
        b1 = (np.sum(df['x']*df['y']) - np.sum(df['x'])*np.sum(df['y'])/len(df))/(np.sum(np.square(df['x']))-np.sum(df['x'])**2/len(df))
        b0 = np.average(df['y']) - b1 * np.average(df['x'])
        estimated_y = b0 + b1 * df['x']
        return b0, b1, estimated_y
    
    # This function calculates the anova table (JUST FOR SIMPLE REGRESSION MODELING)
    def anova_table(n, y, estimated_y, y_avg):
        # ANOVA table
        source_of_variation = pd.Series(['Regression', 'Residuals', 'Total'])

        SSR = np.sum(np.square(estimated_y-y_avg))
        SSE = np.sum(np.square(y - estimated_y))
        SST = np.sum(np.square(y - y_avg))


        degrees_of_freedom = pd.Series([1, n-2, n-1])

        sum_of_squares = pd.Series([SSR, SSE, SST])

        MSR = SSR/1
        MSE = SSE/(n-2)

        median_square = pd.Series([MSR, MSE])

        # Test Statistic
        F0 = pd.Series(MSR/MSE)
        
        # ANOVA TABLE DATAFRAME
        anova_table = pd.DataFrame({'source_of_variation':source_of_variation, 'degrees_of_freedom':degrees_of_freedom, 'sum_of_squares':sum_of_squares, 'median_square':median_square, 'F0':F0})

        # Inverse F Distribution
        F_alpha = stats.f.isf(.05, 1, n-2)

        R_square = SSR/SST
        
        return anova_table, F_alpha, R_square
    
    # This function calculates the confidence intervals of the coefficients for the model (JUST FOR SIMPLE REGRESSION MODELING)
    def confidence_intervals(b0, b1, n, x_avg, MSE, xi):
        # b1 defines as:
        left_interval_b1 = b1 - stats.t.isf(.05/2, n-2) * np.sqrt(MSE/np.sum(np.square(xi - x_avg)))
        right_interval_b1 = b1 + stats.t.isf(.05/2, n-2) * np.sqrt(MSE/np.sum(np.square(xi - x_avg)))

        # b0 defines as:
        left_interval_b0 = b0 - stats.t.isf(.05/2, n-2) * np.sqrt(MSE * (1/n + x_avg**2/np.sum(np.square(xi - x_avg))))
        right_interval_b0 = b0 + stats.t.isf(.05/2, n-2) * np.sqrt(MSE * (1/n + x_avg**2/np.sum(np.square(xi - x_avg))))
        
        return {"li_b0":left_interval_b0, "ri_b0":right_interval_b0, "li_b1":left_interval_b1, "ri_b1":right_interval_b1}
    
    # This function calculates the performance level of the regression (JUST FOR SIMPLE REGRESSION MODELING)
    def performance_level(r_square):
        if r_square > .9:
            return "Very Good"
        elif r_square > .7:
            return "Good"
        elif r_square > .4:
            return "Moderate"
        elif r_square > .2:
            return "Low"
        else:
            return "Null"
    
    # This functions does the significance test (USED FOR SIMPLE AND MULTIPLE REGRESION MODELING)
    def significance_test(F0, F_alpha):
        # SIGNIFICANCE TEST
        # H0: b1 = 0 which means the regression is not significative. This also means the independient variable does not predict the dependient variable
        # Ha: b1 != 0 which means the regression is significative. This also means the independient variable predict the dependient variable

        # We reject the null hipothesis (H0) if F0 > F.05,1,n-2
        if F0 > F_alpha:
            return f"{F0.__round__(4)} > {F_alpha.__round__(4)}, Significant Regression"
        else:
            return f"{F0.__round__(4)} < {F_alpha.__round__(4)}, Insignificant Regression"

# THIS CLASS HAS THE REGRESSION MODELS FOR SIMPLE REGRESSION MODELING (LINEAR, POWER, EXPONENCIAL, LOGARITMIC, AND RECIPROCAL)
class RLS: 
    # LINEAR MODEL
    def linear_model_arg(df, x, y):
        linear_model_data = linearization.linearizable_model(df)

        b0 = linear_model_data[0]
        b1 = linear_model_data[1]
        
        y_avg = df['y'].mean()

        linear_model_anova = linearization.anova_table(len(df), df['y'], linear_model_data[2], y_avg)
        
        anova_table = linear_model_anova[0]
        f_alpha = linear_model_anova[1]
        r_square = linear_model_anova[2]
        
        MSE = linear_model_anova[0].loc[1,'median_square']
        n = len(df)
        x_avg = df['x'].mean()
        
        ci = linearization.confidence_intervals(b0, b1, n, x_avg, MSE,df['x'])
        
        estimated_ecuation = f'{y} = {b0.__round__(4)} + {b1.__round__(4)}*{x}'
        
        residuals = df['y'] - linear_model_data[2]
        
        return {"b0":b0, "b1":b1, "estimated_y":linear_model_data[2], "anova_table":anova_table, "r_square":r_square,"residuals_variance":MSE, "f_alpha":f_alpha, "confidence_intervals":ci, "estimated_ecuation":estimated_ecuation, "residuals":residuals, "forecast_y":linear_model_data[2]}
    
    # POWER MODEL
    def power_model_arg(df, x, y):
        # power model
        lnx = np.log(df['x'])
        lny = np.log(df['y'])

        df_power_model = pd.DataFrame({'x':lnx, 'y':lny})
        power_model_data = linearization.linearizable_model(df_power_model)

        e = np.e
        b0 = e**power_model_data[0]
        b1 = power_model_data[1]

        y_power_model = b0 * df['x']**power_model_data[1]
        power_model_anova = linearization.anova_table(len(df), lny, power_model_data[2], power_model_data[2].mean())
        
        anova_table = power_model_anova[0]
        f_alpha = power_model_anova[1]
        r_square = power_model_anova[2]
        
        MSE = power_model_anova[0].loc[1,'median_square']
        n = len(df)
        x_avg = lnx.mean()
        
        ci = linearization.confidence_intervals(power_model_data[0], b1, n, x_avg, MSE, lnx)
        
        estimated_ecuation = f'{y} = {b0.__round__(4)}*{x}^{b1.__round__(4)}'
        
        residuals = lny - power_model_data[2]
        
        return {"b0":b0, "b1":b1, "estimated_y":y_power_model, "anova_table":anova_table, "r_square":r_square, "residuals_variance":MSE , "f_alpha":f_alpha, "confidence_intervals":ci, "estimated_ecuation":estimated_ecuation, "residuals":residuals, "forecast_y":power_model_data[2]}
    
    # EXPONENCIAL MODEL
    def exp_model_arg(df, x, y):
        lny = np.log(df['y'])

        df_exp_model = pd.DataFrame({'x':df['x'], 'y':lny})
        exp_model_data = linearization.linearizable_model(df_exp_model)

        e = np.e
        b0 = e ** exp_model_data[0]
        b1 = exp_model_data[1]

        y_exp_model = b0 * e ** (b1 * df['x'])
        exp_model_anova = linearization.anova_table(len(df), lny, exp_model_data[2], exp_model_data[2].mean())
        
        anova_table = exp_model_anova[0]
        f_alpha = exp_model_anova[1]
        r_square = exp_model_anova[2]
        
        MSE = exp_model_anova[0].loc[1,'median_square']
        n = len(df)
        x_avg = df['x'].mean()
        
        ci = linearization.confidence_intervals(exp_model_data[0], b1, n, x_avg, MSE, df['x'])
        
        estimated_ecuation = f'{y} = {b0.__round__(4)}*e^({b1.__round__(4)}*{x})'
        
        residuals = lny - exp_model_data[2]
        
        return {"b0":b0, "b1":b1, "estimated_y":y_exp_model, "anova_table":anova_table, "r_square":r_square, "residuals_variance":MSE, "f_alpha":f_alpha, "confidence_intervals":ci, "estimated_ecuation":estimated_ecuation,"residuals":residuals, "forecast_y":exp_model_data[2]}
    
    # LOGARITMIC MODEL
    def log_model_arg(df, x, y):
        lnx = np.log(df['x'])

        df_log_model = pd.DataFrame({'x':lnx, 'y':df['y']})
        log_model_data = linearization.linearizable_model(df_log_model)

        b0 = log_model_data[0]
        b1 = log_model_data[1]

        y_log_model = b0 + b1 * lnx
        log_model_anova = linearization.anova_table(len(df), df['y'], log_model_data[2], log_model_data[2].mean())
        
        anova_table = log_model_anova[0]
        f_alpha = log_model_anova[1]
        r_square = log_model_anova[2]
        
        MSE = log_model_anova[0].loc[1,'median_square']
        n = len(df)
        x_avg = lnx.mean()
        
        ci = linearization.confidence_intervals(b0, b1, n, x_avg, MSE, lnx)
        
        estimated_ecuation = f'{y} = {b0.__round__(4)} + {b1.__round__(4)}*ln({x})'
        
        residuals = df['y'] - log_model_data[2]
        
        return {"b0":b0, "b1":b1, "estimated_y":y_log_model, "anova_table":anova_table, "r_square":r_square, "residuals_variance":MSE, "f_alpha":f_alpha, "confidence_intervals":ci, "estimated_ecuation":estimated_ecuation, "residuals":residuals, "forecast_y":log_model_data[2]}
    
    # RECIPROCAL MODEL
    def rec_model_arg(df, x, y):
        y_rec = 1/df['y']
        x_rec = 1/df['x']

        df_rec_model = pd.DataFrame({'x':x_rec, 'y':y_rec})
        rec_model_data = linearization.linearizable_model(df_rec_model)

        b0 = rec_model_data[0]
        b1 = rec_model_data[1]

        y_rec_model = df['x'] / (b0*df['x'] - b1)
        rec_model_anova = linearization.anova_table(len(df), y_rec, rec_model_data[2], rec_model_data[2].mean())
        
        anova_table = rec_model_anova[0]
        f_alpha = rec_model_anova[1]
        r_square = rec_model_anova[2]
        
        MSE = rec_model_anova[0].loc[1,'median_square']
        n = len(df)
        x_avg = x_rec.mean()
        
        ci = linearization.confidence_intervals(b0, b1, n, x_avg, MSE, x_rec)
        
        estimated_ecuation = f'{y} = {x}/({b0.__round__(4)}*{x}-{b1.__round__(4)})'
        
        residuals = y_rec - rec_model_data[2]
        
        return {"b0":b0, "b1":b1, "estimated_y":y_rec_model, "anova_table":anova_table, "r_square":r_square, "residuals_variance":MSE, "f_alpha":f_alpha, "confidence_intervals":ci, "estimated_ecuation":estimated_ecuation, 'residuals':residuals, "forecast_y":rec_model_data[2]}
    
    # This function calculates the best model
    # The outputs for this functions are a Table with sumarry information about all the models, the best model with it's estimated ecuation, and the regression plot of the model.
    def best_model(df, calculations):
        # Name of the models
        Models = ['Linear', 'Power', 'Exponencial', 'Logaritmic', 'Reciprocal']

        # Estimated ecuation of the models
        est_ec = [ecuation['estimated_ecuation'] for ecuation in calculations]

        # List to present in a percentage format the performance of the models in the linearizable models table
        r_square = [f"{value['r_square']*100:.2f}%" for value in calculations]
        
        # Level of Performance of the models
        performance_level = [linearization.performance_level(i['r_square']) for i in calculations]

        # List to present in 4 decimls the variance of the models in the linearizable models table
        variance = [variance['residuals_variance'].__round__(4) for variance in calculations]
        
        # Confidence Intervals of the Coeficients
        ci_b0 = [f"{i['confidence_intervals']['li_b0'].__round__(4)} < b0 < {i['confidence_intervals']['ri_b0'].__round__(4)}" for i in calculations]
        ci_b1 = [f"{i['confidence_intervals']['li_b1'].__round__(4)} < b1 < {i['confidence_intervals']['ri_b1'].__round__(4)}" for i in calculations]
        
        # Significance Test
        significance_test = [linearization.significance_test(i['anova_table']['F0'][0], i['f_alpha']) for i in calculations]

        # linearizable Models Table
        lmt = pd.DataFrame({"Model":Models, "Estimated Ec.":est_ec, "R^2":r_square, 'Performance Level':performance_level, "Residuals Var":variance,"Significance Test":significance_test, "CI b0":ci_b0, "CI b1":ci_b1})

        # MODEL WITH THE BEST PERFORMANCE
        # List to find the model with the best performance
        r_square_list = [value['r_square'] for value in calculations]

        # Maximum performance of the list of models
        r_square_max = max(r_square_list)

        # Model/s with the best performance (index/es)
        r_square_max_index = [i for i, valor in enumerate(r_square_list) if valor == r_square_max]

        # List to find the model with the lowest residual variance
        residuals_var_list = [value['residuals_variance'] for value in calculations]

        # List to calculate the model with the lowest variance, in case there are 2 or more models with the same max performance
        if len(r_square_max_index) > 1:
            # Variance of the models that have the same performance, and the highest one
            restant_models = [residuals_var_list[i] for i in r_square_max_index]
            # Minimum value of variance the restant models
            min_var = min(restant_models)
            # Indexes of the models with the same variance, being the lowest of the list
            min_var_index = [i for i,valor in enumerate(restant_models) if valor == min_var]
            # BEST MODEL
            # Models/s with the best performance and lowest variance (Ideally it's gonna be one)
            best_model = lmt.iloc[min_var_index].reset_index(drop=True)
            
            # Best Model Regression Graph
            best_model_graph = plots.model_plot(df, calculations[min_var_index[0]])
            
        else:
            # Model with the best performance
            best_model = lmt.iloc[r_square_max_index,:].reset_index(drop=True)
            
            # Best Model Regression Graph
            best_model_graph = plots.model_plot(df, calculations[r_square_max_index[0]])
        
        return {'linearizable_models_table':lmt, 'best_model':best_model, 'best_model_graph':best_model_graph}

# THIS CLASS CONTAINS FUNCTIONS THAT RETURNS PLOTS, FOR BOTH SIMPLE AND MULTIPLE REGRESSION MODELS
class plots:
    # Matrix Plot. (JUST FOR MULTIPLE REGRESSION MODELS)
    def correl_matrix_plot(df):
        # Correlation Matrix
        corr_matrix = df.corr()

        # Transforming data to do a heatmap
        heatmap_df = corr_matrix.iloc[:,::-1].stack().reset_index()

        # Modifying the column names of the new df created
        heatmap_df.columns = ['x', 'y', 'z']

        # Making correlation values percentage
        heatmap_df['z'] = heatmap_df['z'] * 100

        # Doing the heatmap
        corr_heatmap = px.density_heatmap(heatmap_df, x='x', y='y', z='z',
                                        color_continuous_scale=[[0, 'blue'], [0.5, 'gray'], [1, 'red']],
                                        range_color=[-100, 100], text_auto=True, template="plotly_dark")

        # Title Centered
        corr_heatmap.update_layout(title=dict(text="Correlation Matrix", x=0.5))

        # Styling and modifying the tooltip
        corr_heatmap.update_traces(hovertemplate='(%{x}, %{y}) = %{z:.2f}%')

        # Styling text inside the rectangles as percentage
        corr_heatmap.update_traces(texttemplate='%{z:.2f}%')

        # Adding labels to axis
        corr_heatmap.update_layout(coloraxis_colorbar=dict(title='correlation'),
                                xaxis_title='', yaxis_title='')
        
        return corr_heatmap
    
    # Scatter Plot with tendence line with the regression model selected (JUST FOR SIMPLE REGRESSION MODELS)
    # It was used in the plot that contains subplots with all the regressions which is found in the best model option, and also for the select model option, where you only get the plot of the selected model.
    def model_plot(df, model):
        estimated_y = model['estimated_y']
        performance = model['r_square']
        y_name = 'y'
        estimated_y_name = {f'Estimated {y_name}'}

        # Crear el scatter plot con Plotly Express
        fig = px.scatter(df, x='x', y='y', template='plotly_dark', 
                        color='y', hover_data={'x': True, 'y': True})

        # Agregar la línea como una traza de scatter
        fig.add_trace(go.Scatter(x=df['x'], y=estimated_y, mode='lines', name='Estimated Y', line=dict(color='red')))

        # Configurar el título con Plotly Graph Objects
        fig.update_layout(title=dict(text=f"<b>Associated Model with {performance*100:.2f}% Performance: {model['estimated_ecuation']}</b>",
                                    x=0.5,  # Centrado horizontalmente
                                    y=0.95,  # Alineado en la parte superior
                                    xanchor='center',  # Anclaje horizontal al centro
                                    yanchor='top'))  # Anclaje vertical en la parte superior

        # Actualizar las trazas de los marcadores
        fig.update_traces(marker=dict(color='gray', opacity=.6, size=8), hovertemplate='x: %{x}<br>y: %{y}')
        
        return fig
    
    # Function to make a histogram, you have to insert a dataframe with the Intervals o value (Or wathever format you want your "X" axis to have) in the first column, and in the second column the frequencies
    def histogram(df, title):
        x_name = df.columns[0]
        y_name = df.columns[1]
        
        # Crear el histograma con Plotly Express
        fig = px.histogram(df, template='plotly_dark', title=title, x=x_name,y=y_name)
        
        # Actualizar el color de las barras y agregar borde negro
        fig.update_traces(marker=dict(color='blue', line=dict(color='white', width=1)))
        
        # Eliminar la leyenda
        fig.update_layout(showlegend=False)
        
        # Personalizar los nombres de los ejes
        fig.update_layout(
            xaxis_title=x_name,
            yaxis_title=y_name
        )
        
        # Centrar el título principal
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,  # Centrado horizontalmente
                y=0.95,  # Alineado en la parte superior
                xanchor='center',  # Anclaje horizontal al centro
                yanchor='top'  # Anclaje vertical en la parte superior
            ),
            bargap=0
        )
        
        return fig

# This is the official class for the simple regression model outputs
class webAppRegSimple:
    # best model function. Returns a list which contains the best model (With the estimated ecuation), the best model regression plot, a table with all the models summary, and the plot with all the regressions.
    def best_model(df):
        # Name of the models
        model_names = ['Linear', 'Power', 'Exponential', 'Logaritmic', 'Reciprocal']
        
        # List of All th models
        all_models = [RLS.linear_model_arg(df, 'x', 'y'), RLS.power_model_arg(df, 'x', 'y'), RLS.exp_model_arg(df, 'x', 'y'), RLS.log_model_arg(df, 'x', 'y'), RLS.rec_model_arg(df, 'x', 'y')]
        
        # Best Model
        best_model = RLS.best_model(df, all_models)

        # Best Model and its Estimated Ecuation
        best_model_result = f"Best Model: {best_model['best_model']['Model'][0]} Model\n\nEstimated Ecuation: {best_model['best_model']['Estimated Ec.'][0]}"

        # Regrssion Graph of the Best Model
        best_model_graph = best_model['best_model_graph'].update_layout(title=dict(text=f"{best_model['best_model']['Model'][0]} Model with {best_model['best_model']['R^2'][0]} Performance: {best_model['best_model']['Estimated Ec.'][0]}"))
        
        # Table with summarized information of the results of all the models
        summarized_models_table = best_model['linearizable_models_table']
        
        # Model Regression Graphs
        # Crear un subplot con 2 filas y 3 columnas
        fig = make_subplots(rows=2, cols=3)

        # Suponiendo que `all_models` es tu lista de modelos
        # all_models = [model1, model2, model3, model4, model5]
        # `df` es tu DataFrame con los datos

        for i, model in enumerate(all_models, start=1):
            row = (i - 1) // 3 + 1  # Calcular el número de fila (1, 1, 1, 2, 2)
            col = (i - 1) % 3 + 1   # Calcular el número de columna (1, 2, 3, 1, 2)
            
            # Obtener el gráfico para el modelo actual
            subplot = plots.model_plot(df, model)
            
            # Agregar el gráfico al subplot correspondiente
            for trace in subplot.data:
                fig.add_trace(trace, row=row, col=col)
            
            # Ajustar el título para que se alinee con el gráfico correspondiente
            # Usando coordenadas normalizadas
            x_pos = (col - 1) / 3 + 1 / 6  # Coordenadas normalizadas centradas para cada columna
            y_pos = 1.0 if row == 1 else 0.45
            
            # Agregar título al subgráfico
            fig.add_annotation(text=f"{model_names[i-1]} Regression Model", xref="paper", yref="paper",
                            x=x_pos, y=y_pos, xanchor="center", yanchor="bottom", showarrow=False)

        # Quitar la leyenda
        fig.update_layout(showlegend=False)

        # Agregar el título principal
        fig.update_layout(title_text="Regresion Models", title_x=0.5)

        # Aplicar el template 'plotly_dark'
        fig.update_layout(template='plotly_dark')
        
        return best_model_result, best_model_graph, summarized_models_table, fig
    
    # Select Model function. This function returns the estimated ecuation, significance test, r_square, performance level, residuals_variance (Variability of the residuals, also important for selecting a model),
    # Plot of the regression model, coefficient confidence intervals, anova table, the assumptions that the model must meet (Residuals distributed Normal with mean 0 and variance of the residuals, 
    # Incorrelated Data, and constant variance of the residuals), the qqplot, and the atypical values of the sample. 
    def select_model(df, model):
        if model == 'Linear':
            model_data = RLS.linear_model_arg(df, 'x', 'y')
        elif model == 'Power':
            model_data = RLS.power_model_arg(df, 'x', 'y')
        elif model == 'Exponential':
            model_data = RLS.exp_model_arg(df, 'x', 'y')
        elif model == 'Logaritmic':
            model_data = RLS.log_model_arg(df, 'x', 'y')
        elif model == 'Reciprocal':
            model_data = RLS.rec_model_arg(df, 'x', 'y')
        
        estimated_ecuation = model_data['estimated_ecuation']
        significance_test = linearization.significance_test(model_data['anova_table']['F0'][0],model_data['f_alpha'])
        r_square = model_data['r_square']
        performance_level = linearization.performance_level(r_square)
        residuals_variance = model_data['anova_table']['median_square'][1]
        model_graph = plots.model_plot(df,model_data)
        coeficient_intervals = f"Coeficient Condidence Intervals:\n{model_data['confidence_intervals']['li_b0'].__round__(4)} < b0 {model_data['confidence_intervals']['ri_b0'].__round__(4)}\n{model_data['confidence_intervals']['li_b1'].__round__(4)} < b1 {model_data['confidence_intervals']['ri_b1'].__round__(4)}"
        anova_table = model_data['anova_table']
    
        # ASSUMPTIONS
        # CONSTANT VARIANCE
        residuals = model_data['residuals']
        estimated_y = model_data['forecast_y']
        constant_variance_plot = global_functions.scatter_plot(pd.DataFrame({'Residuals':estimated_y, 'Fitted Value':residuals}),'Versus Fits')
        
        # NORMAL DISTRIBUTION MEAN 0
        # H0: Residuals come from a Normal distribution with 0 mean
        # Ha: Residuals come from anothe distribution
        n = len(df)
        classes = (n**(1/2)).__round__()

        # Funcion Para Redondear hacia Arriba
        def redondear_hacia_arriba(numero, decimales=4):
            if isinstance(numero, float):
                parte_decimal = numero - int(numero)
                if parte_decimal > 0:
                    redondeado = math.ceil(numero * 10**decimales) / 10**decimales
                    return round(redondeado, decimales)
            return numero

        width = redondear_hacia_arriba((residuals.max() - residuals.min())/classes)
        linf = [residuals.min()] + [residuals.min() + width*i for i in range(1,classes)]
        lsup = [residuals.min() + width*i for i in range(1,classes)] + [residuals.max()]

        # List of Frequencies by Interval
        frequence = [0]*classes

        # Counting Frequence in each Interval
        for value in residuals:
            for i in range(classes):
                if linf[i] <= value < lsup[i]:
                    frequence[i] += 1
                    break
        frequence[-1] += 1

        # Residuals Standard Deviation
        residuals_std = (model_data['anova_table']['median_square'][1])**(1/2)

        # Cumulative Distribution Function
        cdf = [stats.norm.cdf(lsup[0], scale=residuals_std, loc=0)] + [stats.norm.cdf(lsup[i], scale=residuals_std, loc=0) - stats.norm.cdf(linf[i], scale=residuals_std, loc=0) for i in range(1,classes-1)] + [1-stats.norm.cdf(lsup[classes-2], scale=residuals_std, loc=0)]

        # Expected Frequency
        exp_freq = [i*n for i in cdf]
        
        quotient = [np.square(exp_freq[i] - frequence[i])/exp_freq[i] for i in range(classes)]

        test_statistic = sum(quotient)

        # Chi Square Inverse Functions with alpha = 0.05 and n = classes - 2
        chi_square = stats.chi2.isf(.05, classes-2)

        if test_statistic > chi_square:
            normal_0_mean_hypothesis = f"{test_statistic.__round__(4)} > {chi_square.__round__(4)}, Residuals come from another distribution"
        else:
            normal_0_mean_hypothesis = f"{test_statistic.__round__(4)} < {chi_square.__round__(4)}, Residuals come from a Normal distribution with mean 0"
        
        # QQPLOT: Also for Normal Distribution Asumption
        qqplot = global_functions.qqplot(residuals,residuals_std**2)
        
        # INCORRELATION ASSUMPTION
        # H0: p = 0, which means the analysed data doesn't have correlation
        # Ha: p > 0, which means the analysed data has correlation
        # Reject H0 if d < dL
        # Don't rehect H0 if d > dU
        # Inconclusive test if dL < d < dU
        d = sum(np.square([residuals[i+1] - residuals[i] for i in range(n-1)]))/sum(np.square(residuals))
        
        finding_row = durbin_watson.loc[(durbin_watson['sample_size'] == len(df)) & (durbin_watson['n_terms'] == len(df.columns))].reset_index(drop=True)
        dL = finding_row['DL'][0]
        dU = finding_row['DU'][0]
        
        if d < dL:
            incorrelation_test = f"{d} < {dL}, Correlated data."
        elif d > dU:
            incorrelation_test = f"{d} > {dU}, Incorrelated data."
        else:
            incorrelation_test = f"{dL} < {d} < {dU}, Inconclusive test."
        
        # ATYPICAL DATA
        
        # Residuals Mean
        residuals_mean = np.mean(residuals)
        # Standarized Residuals
        standarized_residuals = [(value-residuals_mean)/(residuals_std) for value in residuals]
        # Finding Atypical Data
        atypical_data = {f'Observation {i+1} with Standarized Value {value}':residuals[i] for i,value in enumerate(standarized_residuals) if abs(value) > 3}
        # If there is not atypical data, it will tell the user
        if len(atypical_data) == 0:
            atypical_data = "There is not atypical data in the sample"
        
        return {'estimated_ec':estimated_ecuation, 'significance_test':significance_test, 'r_square':r_square, 'performance_level':performance_level, 'residuals_variance':residuals_variance, 'model_graph':model_graph, 'coef_intervals':coeficient_intervals, 'anova_table':anova_table, 'constant_variance_plot':constant_variance_plot, 'Normal_0_mean':normal_0_mean_hypothesis, 'qqplot':qqplot, 'incorrelation_test':incorrelation_test, 'atypical_data':atypical_data}

# This is the official class for the correlation data of the SIMPLE REGRESSION MODELS  
class webAppCorrSimple:
    # Correlation Plot. Plot to se how correlated the "X" and "Y" variables are (X is the independient and Y is the dependient variable)
    def correlation_plot(df):
        correlation_plot = global_functions.scatter_plot(df,'Correlation Plot')
        return correlation_plot
    
    # This function will tell you how correlated your data is
    def correlation(df):
        correl = df.corr().iloc[1,0]
        
        if abs(correl) == 1 and correl > 0:
            return f"Correlation: {correl}\nPerfect Positive Correlation"
        elif abs(correl) == 1 and correl < 0:
            return f"Correlation: {correl}\nPerfect Negative Correlation"
        elif abs(correl) > .9 and correl > 0:
            return f"Correlation: {correl}\nVery Strong Positive Correlation"
        elif abs(correl) > .9 and correl < 0:
            return f"Correlation: {correl}\nVery Strong Negative Correlation"
        elif abs(correl) > .7 and correl > 0:
            return f"Correlation: {correl}\nStrong Positive Correlation"
        elif abs(correl) > .7 and correl < 0:
            return f"Correlation: {correl}\nStrong Negative Correlation"
        elif abs(correl) > .4 and correl > 0:
            return f"Correlation: {correl}\nModerate Positive Correlation"
        elif abs(correl) > .4 and correl < 0:
            return f"Correlation: {correl}\nModerate Negative Correlation"
        elif abs(correl) > .2 and correl > 0:
            return f"Correlation: {correl}\nWeak Positive Correlation"
        elif abs(correl) > .2 and correl < 0:
            return f"Correlation: {correl}\nWeak Negative Correlation"
        elif correl > 0:
            return f"Correlation: {correl}\nNegligible Positive Correlation"
        else:
            return f"Correlation: {correl}\nNegligible Negative Correlation"

# Official class of the correlation data for the MULTIPLE REGRESSION MODELS
class webAppCorrMultiple:
    # Correlation heatmap. Plot made from the class "plots"
    def correlation_matrix_heatmap(df):
        plot = plots.correl_matrix_plot(df)
        return plot
    
    # Correlation Matrix Table.
    def correlation_matrix_table(df):
        table = df.corr()
        return table

# Class with calculations for the MULTIPLE REGRESSION MODELS
class RegMultiple:
    # This function returns a list with the following data: y forecast (numpy array), betas (Data Frame), ecuation (string), Cjj (list)
    def ecuation_est(X,Y,regressionToOrigin):
        
        if regressionToOrigin == False:
            X = sm.add_constant(X)

        betas = pd.DataFrame()
        betas["variable"] = X.columns

        X = X.values
        Y = Y.values
        Xt = X.T 
        XtX = np.dot(Xt,X)
        XtX_inv = np.linalg.inv(XtX)
        XtX_inv_Xt = np.dot(XtX_inv,Xt)
        b = np.dot(XtX_inv_Xt,Y)

        betas["beta"] = b
        
        y_est = np.dot(X,b)
        
        if regressionToOrigin == False:
            ecuation = f"Ŷ = {b[0][0].__round__(2)}"
        else:
            ecuation = f"Ŷ ="

        s = {
            0: '₀',
            1: '₁',
            2: '₂',
            3: '₃',
            4: '₄',
            5: '₅',
            6: '₆',
            7: '₇',
            8: '₈',
            9: '₉'
        }
        
        sDef = lambda x: s[x] if x<10 else s[int(str(x)[:1])] + s[int(str(x)[1:])]
        for i,beta in enumerate(b[1:]):
            if beta >= 0:
                ecuation += f" +{beta[0].__round__(2)}X{sDef(i+1)}"
            else:
                ecuation += f" -{-beta[0].__round__(2)}X{sDef(i+1)}"
                
        Cjj = [XtX_inv[i][i] for i in range(len(XtX_inv))]
        
        # For the Hat matrix
        # The formula is the diagonal of X(X'X)^(-1)X'
        H = np.dot(np.dot(X,XtX_inv),Xt)
        hi = H.diagonal()
        
        return y_est,betas,ecuation,Cjj,hi
    
    # This function depends on the "ecuation_est()" function. It returns a dict with the anova table (data frame), r_square (int. which means Variability of the Model for multiple regression), 
    # r_squared_adj (int. which measures the model performance), significance test (str), betas (numpy array), y forecast (numpy array), residuals (numpy array), and the estimated ecuation (str)
    def anova_table(df, y_index, y_name, regToOrigin):
        Y = df.iloc[:,y_index:y_index+1]
        X = df.drop(y_name, axis=1)

        ec_est = RegMultiple.ecuation_est(X,Y,regToOrigin)
        Y_est = ec_est[0]
        residuals = (np.array(Y) - Y_est).T[0]
        
        b = ec_est[1]
        ecuation = ec_est[2]

        n = len(df)
        k = len(X.columns)

        anova_table =pd.DataFrame()

        anova_table['source_of_variation'] = ['Regression','Residuals','Total']
        anova_table['degrees_of_freedom'] = pd.Series([k,n-k-1,n-1])
        
        SSR = sum(np.square(Y_est-Y_est.mean()))[0]
        SSE = sum(np.square(Y.values - Y_est))[0]
        SST = sum(np.square(Y.values - Y_est.mean()))[0]
        anova_table['sum_of_squares'] = pd.Series([SSR,SSE,SST])

        MSR = SSR/k
        MSE = SSE/(n-k-1)
        anova_table['half_square'] = pd.Series([MSR,MSE])
        f0 = MSR/MSE
        anova_table['f0'] = pd.Series([f0])

        f_alpha = stats.f.isf(.05,k,n-k-1)

        significance_test = linearization.significance_test(f0,f_alpha)
        
        r_square = SSR/SST
        
        r_square_adj = 1- (1-r_square) * (n-1)/(n-k-1)
        
        hi = ec_est[4]
        
        return {'anova_table':anova_table,'r_square':r_square,'r_square_adj':r_square_adj,'significance_test':significance_test, 'bi':b, 'y_est':Y_est.T[0],'residuals':residuals, 'est_ecuation':ecuation, 'hi':hi}
    
# Official class for the MULTIPLE REGRESSION MODELING outputs    
class webAppRegMultiple:
    # Will return assumption plots and results, contains also the model data calculated with the "anova_table" function from the "RegMultiple" class
    # TO DEFINE: THE DATA THAT WE WANT TO RETURN FROM THE model_data VARIABLE
    def mult_reg_model(df, regToOrigin):
        y_definition = global_functions.select_y_var(df)
        y_name = y_definition['y_name']
        y_index = y_definition['y_index']
        
        model_data = RegMultiple.anova_table(df,y_index,y_name, regToOrigin)
        
        ## ASSUMPTIONS
        versus_fits_df = pd.DataFrame({'Fitted Value':model_data['y_est'],'Residuals':model_data['residuals']})
        versus_order_df = pd.DataFrame({'Observation Order':[i for i in range(1,len(df) + 1)],'Residuals':model_data['residuals']})

        # RESIDUALS CONSTANT VARIANCE
        # Versus Fits Plot
        versus_fit_plt =global_functions.scatter_plot(versus_fits_df,'Versus Fits')

        # INCORRELATION OF THE RESIDUALS
        # Versus Order Plot
        versus_order_plt = global_functions.scatter_plot(versus_order_df,'Versus Order')
        
        # NORMALITY OF THE RESIDUALS
        # To visually analyze it we can use the qqplot
        normal_prob_plt = global_functions.qqplot(model_data['residuals'],model_data['anova_table']['half_square'][1])
        
        # Normality mean 0 variance of the residuals
        normal_0_mean_asumption, normal_0_mean_hist =global_functions.normality_0_mean(model_data['anova_table'],model_data['residuals'])
        
        # INCORRELATED RESIDUALS ASSUMPTION
        # For this test we're going to need the sample size "n" (the function calculates it), number of independient variables "k" and residuals
        incorrel_res_assumption = global_functions.incorrelation_assumption(model_data['residuals'],len(df.columns) - 1)
        
        # ATYPICAL DATA
        # The parameters to use the function are the residuals, the variance of the residuals, and the Hat values
        atypical_data = global_functions.atypical_data(model_data['residuals'],model_data['anova_table']['half_square'][1], model_data['hi'])
        
        return {'versus_fit':versus_fit_plt,'versus_order':versus_order_plt,'qqplot':normal_prob_plt,'normal_0_mean_assumption':normal_0_mean_asumption,'normal_0_mean_hist':normal_0_mean_hist,'incorrel_res_assumption':incorrel_res_assumption,'atypical_data':atypical_data,'model_data':model_data}
    
    # Returns a data frame with a summary for the coefficients (name of the terms, value of the coefficients, confidence interval for the coefficients and its VIF)
    # This table is essential at the begining, where you run the model with all the variables of the sample. If you have a VIF higher than 10, it is recommended to eliminate the variable with the higher VIF,
    # or to expand the sample size.
    def coef_summary(df, regToOrigin):
        # Function for selecting the "y" column in the dataframe
        y_definition = global_functions.select_y_var(df)
        y_name = y_definition['y_name']
        y_index = int(y_definition['y_index'])

        # Dataframe for "X" variables and "Y" variable columns
        X = df.drop(y_name, axis=1)
        
        if regToOrigin == False:
            X = sm.add_constant(X)
        Y = df.iloc[:,y_index:y_index+1]

        # VIF
        # VIF formula is VIFj = 1/(1-Rj^2)
        # Using the anova table function to determine Rj^2 which is the multiple determination coefficient obtained by doing the regression "Xi" on the other regresion variables.
        models_data = [RegMultiple.anova_table(X, index, name, regToOrigin) for index,name in enumerate(X.columns)]
        # Variability of the estimated regressions (r squared)
        r_square = [i['r_square'] for i in models_data]
        # VIF
        VIF = [1/(1-i) for i in r_square]
        
        #95% CONFIDENCE INTERVALS
        # Bj_estimated - t.inv(alpha/2,n-p)*(res_variance*Cjj) < Bj < Bj_estimated + t.inv(alpha/2,n-p)*(res_variance*Cjj)
        # Cjj is the element of the diagonal (X'X)^-1
        # p is the total number of parameters in the regression model
        
        # Annova table for the estimated ecuation
        anova_est_ec = RegMultiple.anova_table(df,y_index,y_name,regToOrigin)
        # Variance of the Residuals of the estimated ecuation
        residuals_var = anova_est_ec['anova_table']['half_square'][1]
        
        # Ec Estimation data
        ec_est_data = RegMultiple.ecuation_est(X,Y,regToOrigin)
        # Coefficients (b0,b1,..,bk)
        bi = ec_est_data[1]

        # Cjj (Inverse of (XtX)^-1)
        Cjj = ec_est_data[3]
        
        # Confidence Intervals for betas
        n = len(df)
        p = len(df.columns)
        t_alpha = stats.t.isf(.05/2,n-p)
        
        CI_LT = [beta-t_alpha*np.sqrt(Cjj[i]*residuals_var) for i,beta in enumerate(bi['beta'])]
        CI_RI = [beta+t_alpha*np.sqrt(Cjj[i]*residuals_var) for i,beta in enumerate(bi['beta'])]

        confidence_intervals = [f"{CI_LT[i].__round__(4)} < b{i} < {CI_RI[i].__round__(4)}" for i in range(len(CI_LT))]
        
        coefficients_summary = pd.DataFrame()
        coefficients_summary['Term'] = X.columns
        coefficients_summary['Coef'] = bi['beta']
        coefficients_summary['95% CI'] = confidence_intervals
        coefficients_summary['VIF'] = VIF
        
        return coefficients_summary
    
    # This function will return a dataframe with the top 2 models for all of the different sizes of the independient variables (top 2 for a model with 1 independient variable, 2, 3, ..., k)
    # It will show which variables make up each model, with its performance (r_squared_adj) and its variability (r_squared)
    def top_models(df, regToOrigin):
        # Function for selecting the "y" column in the dataframe
        y_definition = global_functions.select_y_var(df)
        y_name = y_definition['y_name']
        y_index = int(y_definition['y_index'])

        # Dataframe for "X" variables and "Y" variable columns
        X = df.drop(y_name, axis=1)
        Y = df.iloc[:,y_index:y_index+1]
        
        variables = X.columns
        all_combinations = []
        r_square = []
        r_square_adj = []
        S = []

        df_x = []
        for i in range(1,len(X.columns)+1):
            comb = combinations(X.columns,i)
            for c in comb:
                all_combinations.append(c)
                df_x.append(df.loc[:,c])
        df_for_regression = [pd.concat([Y, X_vars], axis=1) for X_vars in df_x]
        model = [RegMultiple.anova_table(df,y_index,y_name,regToOrigin) for df in df_for_regression]
        [r_square.append(m['r_square']) for m in model]
        [r_square_adj.append(m['r_square_adj']) for m in model]
        [S.append(m['anova_table']['half_square'][1]) for m in model]
        
        rows = []      
        for comb in all_combinations:
            row = ['X' if var in comb else '' for var in variables]
            row.insert(0,len(comb))
            rows.append(row)

        results_df = pd.DataFrame(rows,columns=['Vars'] + X.columns.tolist())
        results_df['R sq'] = np.round(np.array(r_square)*100,4)
        results_df['R sq (adj)'] = np.round(np.array(r_square_adj)*100,4)
        results_df['S'] = np.round(np.array(S)**.5,4)

        # Ordenar el DataFrame por 'num_vars' y luego por 'R sq'
        df_sorted = results_df.sort_values(by=['Vars', 'R sq (adj)'], ascending=[True, False])

        # Seleccionar el top 2 de cada grupo de 'num_vars'
        top_n_per_group = df_sorted.groupby('Vars').head(2).reset_index(drop=True)
        
        return top_n_per_group