import requests
import pandas as pd
from datetime import datetime as dt 
from functools import reduce
import json

def extraerPIB(indicator_id, name):
    # Token de la API
    token='6ed1aedb-7a89-ced4-fc4c-daf62a952edf'

    # Indicador de PIB (736181 en este caso) y URL para obtener datos históricos
    url = f'https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/{indicator_id}/es/0700/false/BIE/2.0/{token}?type=json'
    
    response = requests.get(url)

    if response.status_code == 200:
        print(f"Sucessfully imported data: {name}")
        data = response.json()
            
        return data
    else:
        print(f"Error {response.status_code}")
        return response.status_code
    
def pibPorActividad():
    actividades = {
            494098: "PIB Nacional",
            494103: "Agricultura",
            494109: "Minería",
            494114: "Generación, Transmisión y Distribución de Energía Eléctrica",
            494116: "Construcción",
            494121: "Industria Alimentaria",
            494131: "Industria de las Bebidas y del Tabaco",
            494191: "Fabricación de Maquinaria y Equipo",
            494158: "Fabricación de Productos Químicos",
            494227: "Comercio al Por Mayor",
            494244: "Servicios Financieros y de Seguros",
            494228: "Transporte y Almacenamiento",
            494254: "Servicios de Salud y Asistencia Social",
            494253: "Servicios Educativos",
            494262: "Servicios de Alojamiento Temporal",
            494265: "Servicios de Reparación y Mantenimiento"
        }

    indicadores = ''
    for i, val in enumerate(actividades.keys()):
        indicadores += f'{val}' if i == 0 else f',{val}'
    name = "pibByActivity"

    data = extraerPIB(indicadores, name)

    dfCleaned = None
    for i, val in enumerate(data['Series']):
        if i == 0:
            dfCleaned = pd.DataFrame(val['OBSERVATIONS'])[['TIME_PERIOD', 'OBS_VALUE']]
            act = actividades[int(val['INDICADOR'])]
            
            dfCleaned.columns = ['Año/Trimestre', act]
            dfCleaned[act] = dfCleaned[act].astype(float)
        else:
            dfAct = pd.DataFrame(val['OBSERVATIONS'])[['OBS_VALUE']]
            
            act = actividades[int(val['INDICADOR'])]
            dfAct.columns = [act]
            dfAct[act] = dfAct[act].astype(float)
            
            dfCleaned = pd.concat([dfCleaned, dfAct], axis=1)
    dfCleaned = dfCleaned.drop_duplicates(subset='Año/Trimestre', keep='first', ignore_index=True).dropna(axis=0)
    
    return dfCleaned

def importacionDatos():
    def dataImport():
        # PIB GENERAL, información trimestral
        indicator_id = '736181'
        indicador_sector_primario = '736195'
        indicador_sector_secundario = '736202'
        indicador_sector_terciario = '736237'

        dataGlobal = extraerPIB(indicator_id, 'PIB_global')
        dataPrimario = extraerPIB(indicador_sector_primario, 'PIB_primarias')
        dataSecundario = extraerPIB(indicador_sector_secundario, 'PIB_secundarias')
        dataTerciario = extraerPIB(indicador_sector_terciario, 'PIB_terciarias')

        dfGlobal = pd.DataFrame(dataGlobal['Series'][0]['OBSERVATIONS'])[['TIME_PERIOD', 'OBS_VALUE']]
        dfPrimario = pd.DataFrame(dataPrimario['Series'][0]['OBSERVATIONS'])[['TIME_PERIOD', 'OBS_VALUE']]
        dfSecundario = pd.DataFrame(dataSecundario['Series'][0]['OBSERVATIONS'])[['TIME_PERIOD', 'OBS_VALUE']]
        dfTerciario = pd.DataFrame(dataTerciario['Series'][0]['OBSERVATIONS'])[['TIME_PERIOD', 'OBS_VALUE']]
        
        dfGlobal['OBS_VALUE'] = dfGlobal['OBS_VALUE'].astype(float)
        dfPrimario['OBS_VALUE'] = dfPrimario['OBS_VALUE'].astype(float)
        dfSecundario['OBS_VALUE'] = dfSecundario['OBS_VALUE'].astype(float)
        dfTerciario['OBS_VALUE'] = dfTerciario['OBS_VALUE'].astype(float)
        
        return {'PIB_global':dfGlobal, 'PIB_primarias':dfPrimario, 'PIB_secundarias':dfSecundario, 'PIB_terciarias':dfTerciario}
    
    data = dataImport()
    
    for i in data:
        data[i].columns = ['Año/Trimestre', i]
        
    df = reduce(lambda left, right: pd.merge(left, right, on='Año/Trimestre', how='outer'), [data[i] for i in data])
    today = dt.today().strftime('%Y-%m-%d')
    filename = 'pibData.json'
    
    try:
        with open(filename, 'w') as json_file:
            json.dump({today:df.to_dict()}, json_file, indent=4)
            print("Data Successfully updated!")
    except Exception as e:
        print(f"An error ocurred: {e}")
    
    return df

def importModules():
    # Importación de clases provenientes de un módulo de un repositorio de Github, que contienen funciones, relacionadas al modelo  de regresión lineal
    # que construiremos en este proyecto.
    import importlib.util
    from cryptography.fernet import Fernet
    import pickle
    
    with open('config.pkl', 'rb') as file:
        url = pickle.load(file)
    
    cipher_suite = Fernet(b'nvymANnm6QIMgArYmcSFTaBDTERJRddf4_85xuKlzHc=')
    moduleLink = cipher_suite.decrypt(url).decode()

    # Download and load the Python file as a module
    response_py = requests.get(moduleLink)
    module_name = "reg_lin_sim_mod"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(response_py.text, module.__dict__)

    # Import the specific functions you need
    cm = module.webAppCorrMultiple
    mlr = module.webAppRegMultiple
    
    return cm, mlr