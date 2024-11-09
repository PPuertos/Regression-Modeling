from plotly import graph_objects as go
import numpy as np

def pibPorSectorBar(df):
    fig = go.Figure()

    fig.add_trace(go.Bar(x=['PIB_primarias', 'PIB_secundarias', 'PIB_terciarias'], y=df[['PIB_primarias', 'PIB_secundarias', 'PIB_terciarias']].iloc[-1], text=[f"<b>${i/1000:,.2f}k</b>" for i in df[['PIB_primarias', 'PIB_secundarias', 'PIB_terciarias']].iloc[-1]]))
    fig.update_layout(title=dict(text=f"<b>Contribución de PIB Nacional de los diferentes sectores</b>", x=.5))
    
    return fig
    
def pibPorSectorProporcion(df):
    pie = go.Figure()
    pie1 = go.Pie(labels=['PIB_primarias', 'PIB_secundarias', 'PIB_terciarias'], values=df[['PIB_primarias', 'PIB_secundarias', 'PIB_terciarias']].iloc[-1], hole=.35)

    pie.add_trace(pie1)
    pie.update_traces(hoverinfo='label+value+percent', textinfo='percent', textfont_size=12,
                    marker=dict(line=dict(color='#000', width=1)))

    pie.add_annotation(text=f"${df['PIB_global'].iloc[-1]/1000000:,.2f}M", x=.5, y=.5, font_size=18, showarrow=False, font=dict(color='white'))
    pie.update_layout(title=dict(text='<b>PIB Por Sector</b>', x=.5))
    
    return pie

def OLSSector(df):    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['Año/Trimestre'], y=df['PIB_global'], mode='lines', name=f'{df.columns[1]}'))

    # Personalizar el layout
    fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Valores',
        legend_title='Leyenda',
        template='plotly'
    )

    Y = df['PIB_global']
    X = df.iloc[:,2:]
    X.insert(0, 'Const', 1)


    Xt = X.T
    XtX_inv = np.linalg.inv(Xt @ X)
    b = XtX_inv @ Xt @ Y

    funcion = f"Function: {b[0]:,.2f}"
    for i in range(len(b[1:])):
        signDet = lambda x, i: f" +{x:,.2f}X{i}" if x>0 else f" {x:,.2f}X{i}"
        
        funcion += signDet(b[i+1], i)

    yEst = X @ np.array(b)

    fig.add_trace(go.Scatter(x=df['Año/Trimestre'], y=yEst, name='Ŷ')).update_layout(title=dict(text=funcion, x=.5))
    
    return fig

def boxPlot(pibPorSector):
    tasaDeRendimiento = pibPorSector.iloc[:,2:].pct_change().dropna(ignore_index=True)

    # Rendimientos Esperados y Desviación Estándar de los rendimientos, anualizados
    rendEsp = tasaDeRendimiento.mean()*4
    desvEst = tasaDeRendimiento.std(ddof=1)*2
    icInferior = rendEsp - desvEst
    icSuperior = rendEsp + desvEst

    import plotly.graph_objects as go
    import pandas as pd

    df = pd.DataFrame({'Sector':rendEsp.index, 'Rendimientos Esperados':rendEsp, 'Desviación Estándar':desvEst, 'IC Inferior':icInferior, 'IC Superior':icSuperior}).reset_index(drop=True)

    # Crear el boxplot
    fig = go.Figure()

    for _, row in df.iterrows():
        fig.add_trace(go.Box(
            y=[row['IC Inferior'], row['Rendimientos Esperados'], row['IC Superior']],
            name=row['Sector'],
            boxpoints=False,  # Sin puntos individuales
            line=dict(color='blue'),
            fillcolor='lightblue',
            hoverinfo="none"
        ))

        # Etiqueta para el rendimiento esperado (arriba de la caja)
        fig.add_annotation(
            x=row['Sector'], y=row['Rendimientos Esperados'],
            text=f"Rendimiento: {row['Rendimientos Esperados']*100:,.2f}%",
            showarrow=False,
            yshift=15  # Ajusta para que esté afuera, encima de la caja
        )

        # Etiqueta para el IC Inferior (debajo de la caja)
        fig.add_annotation(
            x=row['Sector'], y=row['IC Inferior'],
            text=f"IC Inferior: {row['IC Inferior']*100:,.2f}%",
            showarrow=False,
            yshift=-15  # Ajusta para que esté afuera, debajo de la caja
        )

        # Etiqueta para el IC Superior (encima de la caja, pero más alto que el rendimiento)
        fig.add_annotation(
            x=row['Sector'], y=row['IC Superior'],
            text=f"IC Superior: {row['IC Superior']*100:,.2f}%",
            showarrow=False,
            yshift=30  # Más arriba para diferenciar del rendimiento esperado
        )

    fig.update_layout(
        title=dict(text="Rendimiento Esperado e Intervalos de Confianza de Cada Sector", x=.5),
        yaxis_title="Rendimiento Anualizado (%)",
        xaxis_title="Sector",
        showlegend=False
    )

    return fig

def olsVsRealValues(df, betas, estEcuation):
    xDefinitiva = df.iloc[:,2:]
    xDefinitiva.insert(0, 'Const', 1)
    fig = go.Figure()
    lineEst = go.Scatter(x=df['Año/Trimestre'].iloc[::-1], y=(np.array(xDefinitiva)@np.array(betas['beta'])).tolist()[::-1], mode='lines', name='Ŷ')
    line = go.Scatter(x=df['Año/Trimestre'].iloc[::-1], y=df['PIB Nacional'].iloc[::-1], mode='lines', name='PIB Nacional')
    fig.add_traces([line, lineEst]).update_layout(hovermode='x unified', title=dict(text=f"PIB Nacional Estimado : <b>{estEcuation[4:]}</b>", x=.5))
    
    return fig

def donutChartInvRecommendations(betas):
    inversionesRecomendadas = betas.loc[betas['beta'] > 0,:]


    invFig = go.Figure()
    pieChart = go.Pie(labels=inversionesRecomendadas['variable'], values=inversionesRecomendadas['beta'], hole=.35)

    invFig.add_trace(pieChart).update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=11, marker=dict(line=dict(color='#000', width=1))).update_layout(height=600, title=dict(text="<b>Participación de Inversión en Sectores Clave para el PIB Nacional</b>", x=.5))
    
    return invFig