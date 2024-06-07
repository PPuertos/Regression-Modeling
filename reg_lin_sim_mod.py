import pandas as pd
import plotly.express as px
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# The Critical Values of the Durbin Watson Test are from:
# https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/regression/supporting-topics/model-assumptions/test-for-autocorrelation-by-using-the-durbin-watson-statistic/
durbin_watson = pd.read_csv('critical_values_durbin_watson.csv')

class global_functions:
    def select_y_var(df):
        [print(f"{i+1}. {k}") for i,k in enumerate(df.columns)]
        x = input("Select y variable:")
        y_index = int(x)-1
        y_name = df.iloc[:,y_index:y_index+1].columns[0]
        return {'y_index':int(x)-1,'y_name':y_name}

class linearization:
    def linearizable_model(df):
        # LINEAR MODEL: y = b0 + b1 * x
        # ln(y) = ln(b0) + ln(b1) + ln(x)
        # Values of the constants of the linear model
        b1 = (np.sum(df['x']*df['y']) - np.sum(df['x'])*np.sum(df['y'])/len(df))/(np.sum(np.square(df['x']))-np.sum(df['x'])**2/len(df))
        b0 = np.average(df['y']) - b1 * np.average(df['x'])
        estimated_y = b0 + b1 * df['x']
        return b0, b1, estimated_y

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
    
    def confidence_intervals(b0, b1, n, x_avg, MSE, xi):
        # b1 defines as:
        left_interval_b1 = b1 - stats.t.isf(.05/2, n-2) * np.sqrt(MSE/np.sum(np.square(xi - x_avg)))
        right_interval_b1 = b1 + stats.t.isf(.05/2, n-2) * np.sqrt(MSE/np.sum(np.square(xi - x_avg)))

        # b0 defines as:
        left_interval_b0 = b0 - stats.t.isf(.05/2, n-2) * np.sqrt(MSE * (1/n + x_avg**2/np.sum(np.square(xi - x_avg))))
        right_interval_b0 = b0 + stats.t.isf(.05/2, n-2) * np.sqrt(MSE * (1/n + x_avg**2/np.sum(np.square(xi - x_avg))))
        
        return {"li_b0":left_interval_b0, "ri_b0":right_interval_b0, "li_b1":left_interval_b1, "ri_b1":right_interval_b1}
    
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
    
    def significance_test(F0, F_alpha):
        # SIGNIFICANCE TEST
        # H0: b1 = 0 which means the regression is not significative. This also means the independient variable does not predict the dependient variable
        # Ha: b1 != 0 which means the regression is significative. This also means the independient variable predict the dependient variable

        # We reject the null hipothesis (H0) if F0 > F.05,1,n-2
        if F0 > F_alpha:
            return f"{F0.__round__(4)} > {F_alpha.__round__(4)}, Significant Regression"
        else:
            return f"{F0.__round__(4)} < {F_alpha.__round__(4)}, Insignificant Regression"
   
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
    # Returns a list with the following arguments: (b0, b1, yi_estimated, anova_table)
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
    
class plots:
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
                                        color_continuous_scale=[[0, 'darkblue'], [0.5, 'white'], [1, 'darkred']],
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

class webAppRegSimple:
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
        constant_variance_plot = webAppCorrSimple.correlation_plot(pd.DataFrame({'x':estimated_y, 'y':residuals}))

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
        residuals_ordered = sorted(residuals)
        k = [(i-.375)/(n+.25) for i in range(1,n+1)]
        Zk = [stats.norm.ppf(q=i, loc=0, scale=1) for i in k]
        exp_values = [i*residuals_std for i in Zk]
        
        qqplot = webAppCorrSimple.correlation_plot(pd.DataFrame({'x':residuals_ordered, 'y':exp_values}))
        
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
    
class webAppCorrSimple:
    def correlation_plot(df):

        # Crear el scatter plot con Plotly Express
        fig = px.scatter(df, x='x', y='y', template='plotly_dark', 
                        color='y', hover_data={'x': True, 'y': True}, trendline='ols',trendline_color_override='white')


        # Configurar el título con Plotly Graph Objects
        fig.update_layout(title=dict(text=f"<b>Correlation Scatter Plot ({df.corr()['x']['y'].__round__(4)*100:.2f}%)</b>",
                                    x=0.5,  # Centrado horizontalmente
                                    y=0.95,  # Alineado en la parte superior
                                    xanchor='center',  # Anclaje horizontal al centro
                                    yanchor='top'))  # Anclaje vertical en la parte superior

        # Actualizar las trazas de los marcadores
        fig.update_traces(marker=dict(color='blue', size=8), hovertemplate='x: %{x}<br>y: %{y}', line=dict(dash='dot'))
        
        return fig
    
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
        
class webAppCorrMultiple:
    def correlation_matrix_heatmap(df):
        plot = plots.correl_matrix_plot(df)
        return plot

    def correlation_matrix_table(df):
        table = df.corr()
        return table

class RegMultiple:
    def VIF(X):
        X = sm.add_constant(X)  # Añadir constante para el intercepto del modelo

        # Calcular el VIF para cada característica
        vif = pd.DataFrame()
        vif["Variable"] = X.iloc[:,1:].columns
        vif["VIF"] = [variance_inflation_factor(X,i) for i in range(1,len(X.columns))]
        return vif
    
    def ecuation_est(X,Y):
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
        
        return y_est, betas
    
class webAppRegMultiple:
    def anova_table(df):
        y_definition = global_functions.select_y_var(df)
        y_name = y_definition['y_name']
        y_index = y_definition['y_index']
        Y = df.iloc[:,y_index:y_index+1]
        X = df.drop(y_name, axis=1)

        ec_est = RegMultiple.ecuation_est(X,Y)
        Y_est = ec_est[0]
        b = ec_est[1]

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
        
        return {'anova_table':anova_table,'r_square':SSR/SST,'significance_test':significance_test, 'bi':b, 'y_est':Y_est}