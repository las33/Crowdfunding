from deslib.util.instance_hardness import kdn_score
import numpy as np
import shap

def get_hard_easy_instance(X_train, y_train, k = 5):
    kdn_list = kdn_score(X_train.values,y_train.values, 5)
    maximum = np.max(kdn_list[0])
    minimum = np.min(kdn_list[0])
    print(maximum, minimum)

    index_of_maximum = np.where(kdn_list[0] == maximum)
    index_of_minimum = np.where(kdn_list[0] == minimum)

    print(index_of_maximum[0][0], index_of_minimum[0][0])
    hard = X_train.values[index_of_maximum[0][0]]
    easy = X_train.values[index_of_minimum[0][0]]
    
    return index_of_maximum[0][0], index_of_minimum[0][0]

def print_shap(model, X_train,y_train,X_test,y_test,y_predict):
    shap.initjs()

    index_of_maximum, index_of_minimum = get_hard_easy_instance(X_test, y_test)
    
    print(index_of_maximum, index_of_minimum)
    print(y_test.iloc[index_of_maximum], y_test.iloc[index_of_minimum])
    print(y_predict[index_of_maximum], y_predict[index_of_minimum])

    data = shap.sample(X_test, 20)     


    explainer = shap.KernelExplainer(model.predict_proba, data)
    shap_values = explainer.shap_values(X_test)

    #print(explainer.expected_value)
    #print(shap_values.shape)
    #print(X_train.iloc[0,:])       

    f=shap.force_plot(explainer.expected_value[0], shap_values[0][index_of_maximum,:], X_test.iloc[index_of_maximum,:], show=False)
    shap.save_html("index_max.htm", f)

    f=shap.force_plot(explainer.expected_value[0], shap_values[0][index_of_minimum,:], X_test.iloc[index_of_minimum,:], show=False)
    shap.save_html("index_min.htm", f)
    
    #data = shap.sample(X_train, 10) 
    
    #explainer = shap.KernelExplainer(model.predict, data)
    
    #sape_values = explainer.shap_values(X_train)   
    
    
    #shap.dependence_plot("# of Pledges",
    #                 shap_values,
    #                 features=X_train)
    
    #shap.summary_plot(shap_values,
    #              features = X_train)