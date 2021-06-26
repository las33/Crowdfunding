import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def read_and_processing_data(filename):

    df = pd.read_csv(filename)
    print('columns before processing:\n', df.columns)  

    remove_cols = ['DayW']
    df.drop(remove_cols, axis=1, inplace=True)

    df['Macrocategory'] = df.Macrocategory.astype('category')

    months = {1:'Jan', 2:'Feb',3:'Mar',4: 'Apr',5: 'May' ,6:'June',7: 'July',8: 'Aug',9: 'Sept',10:'Oct',11: 'Nov',12: 'Dec'}
    df = df.replace({"Month": months})

    df['Month'] = df.Month.astype('category')
    df['Year'] = df.Year.astype('int')

    df = pd.get_dummies(df, columns=["Macrocategory"])
    df = pd.get_dummies(df, columns=["Month"])        
            

    df = df.rename(columns={'LnPledges': '# of Pledges', 'LnGoal': 'Goal'})
    df = df.rename(columns={'Illiteracy': '% of Illiteracy', 'Rewards': '# of Rewards'})
    df = df.rename(columns={'Popold':'Elderly population', 'LnPopEstm2015': 'Population'})
    df = df.rename(columns={'LnAreaKM':'City area', 'Macrocategory_Arq_Urb': 'Architecture'})
    df = df.rename(columns={'NSSt': 'SMNS' , 'NSSj': 'MMNS'})

    df = df.rename(columns={'Macrocategory_Quadrinhos': 'Comic cartoons', 'LnPIBpercap': 'GDP per capita'})
    df = df.rename(columns={'Macrocategory_Dança': 'Dance', 'Macrocategory_Arte': 'Art'})
    df = df.rename(columns={'Macrocategory_Carnaval': 'Carnival', 'Macrocategory_Comunidade': 'Community'})
    df = df.rename(columns={'Macrocategory_Circo': 'Circus', 'Macrocategory_Design': 'Design'})
    df = df.rename(columns={'Macrocategory_Educação': 'Education', 'Macrocategory_Esporte': 'Sport'})
    df = df.rename(columns={'Macrocategory_Eventos': 'Events', 'Macrocategory_Fotografia': 'Photography'})
    df = df.rename(columns={'Macrocategory_Gastronomia': 'Gastronomy', 'Macrocategory_Jogos': 'Games'})
    df = df.rename(columns={'Macrocategory_Jornalismo': 'Journalism', 'Macrocategory_Literatura': 'Literature'})
    df = df.rename(columns={'Macrocategory_Meio Ambiente': 'Environment'})
    df = df.rename(columns={'Macrocategory_Moda': 'Fashion', 'Macrocategory_Música': 'Music'})
    df = df.rename(columns={'Macrocategory_Negócios Sociais': 'Social Business', 'Macrocategory_Teatro': 'Theater'})
    df = df.rename(columns={'Macrocategory_Mob_Transporte':'Transportation'})    
    df = df.rename(columns={'Macrocategory_C&T':'R&D', 'Macrocategory_Cin_Video': 'Movie'})    

    df = df.rename(columns={'Dom1': 'Sunday', 'Qua4': 'Wednesday'})
    df = df.rename(columns={'Seg2': 'Monday', 'Qui5': 'Thursday'})
    df = df.rename(columns={'Ter3': 'Tuesday', 'Sex6': 'Friday'})
    df = df.rename(columns={'Sab7': 'Saturday'})


    df = df.rename(columns={'Month_Apr' : 'April', 'Month_Jan': 'January'})
    df = df.rename(columns={'Month_Aug': 'August', 'Month_July' : 'July' })
    df = df.rename(columns={'Month_Dec': 'December', 'Month_June': 'June'})
    df = df.rename(columns={'Month_Feb': 'February', 'Month_Mar': 'March'})
    df = df.rename(columns={'Month_May': 'May', 'Month_Oct' : 'October' })
    df = df.rename(columns={'Month_Nov': 'November', 'Month_Sept': 'September'})

    print('columns after processing:\n', df.columns)   

    return df

def get_Best_Features(df):
    y = df.Success
    X = df.drop(labels=['Success', 'Year'], axis=1)
    ss = StandardScaler()
    X = pd.DataFrame(ss.fit_transform(X),columns = X.columns)
    model = RandomForestRegressor()
    model.fit(X,y)
    names = list(X)
    importances = model.feature_importances_
    
    features_score = []
    
    for i,v in enumerate(importances): 
        features_score.append([names[i],v])
    
    features_score.sort(key=lambda x: x[1],reverse=True)
    
    for f in features_score:
        print(f)

def scaler_set(df):
    ss = StandardScaler()
    df = pd.DataFrame(ss.fit_transform(df),columns = df.columns)

    return df

