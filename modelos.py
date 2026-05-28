from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


def executar_random_forest(X_train, X_test, y_train, y_test, pesos_classe):
    print("\n[ALGORITMO] Treinando Random Forest Otimizado (Com Poda)...")
    
    # Aplicando regularização (poda) para evitar que o modelo decore o ruído
    modelo = RandomForestClassifier(
        n_estimators=150,        # Aumentamos o comitê para maior estabilidade
        max_depth=8,             # PODA: Impede que as árvores fiquem muito profundas e decorem detalhes
        min_samples_split=10,    # Exige pelo menos 10 horas de exemplo para tomar uma decisão nova
        min_samples_leaf=5,      # Garante folhas finais mais generalistas
        class_weight=pesos_classe, 
        random_state=42,
        n_jobs=-1
    )
    
    modelo.fit(X_train, y_train)
    print("Treinamento do Random Forest Otimizado concluído!")
    
    print("Avaliando com Limiar de Decisão Ajustado (Threshold: 55%)...")
    # Em vez de prever direto, pegamos a probabilidade matemática de cada cenário
    probabilidades = modelo.predict_proba(X_test)
    
    # probabilidades[:, 1] nos dá a certeza do modelo de que o mercado vai SUBIR
    # Só prevemos 1 (Alta) se o modelo tiver mais de 55% de certeza. Caso contrário, assume Queda (0).
    limiar = 0.55
    y_pred = (probabilidades[:, 1] >= limiar).astype(int)
    
    exibir_metricas("RANDOM FOREST OTIMIZADO", y_test, y_pred)
    return modelo


def executar_logistica_l1(X_train, X_test, y_train, y_test, pesos_classe):
    print("\n[ALGORITMO] Inicializando Regressão Logística L1 OTIMIZADA...")
    
    print("Aplicando padronização de recursos (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ajuste fino de parâmetros para calar o Warning do Scikit-Learn 1.8+
    # No modo moderno, definimos l1_ratio=1.0 diretamente sem passar a string 'penalty'
    modelo = LogisticRegression(
        solver='saga',         
        l1_ratio=1.0,          # Força 100% Lasso (L1) matematicamente
        class_weight=pesos_classe,
        random_state=42,
        C=0.7,                 
        max_iter=3000          
    )
    
    print("Treinando o modelo estatístico linear (SAGA)...")
    modelo.fit(X_train_scaled, y_train)
    print("Treinamento concluído!")
    
    print("Aplicando Limiar de Decisão Ajustado (Threshold: 54%)...")
    probabilidades = modelo.predict_proba(X_test_scaled)
    limiar = 0.54
    y_pred = (probabilidades[:, 1] >= limiar).astype(int)
    
    exibir_metricas("REGRESSÃO LOGÍSTICA L1 OTIMIZADA", y_test, y_pred)
    
    coeficientes_zerados = np.sum(modelo.coef_[0] == 0)
    total_recursos = X_train.shape[1]
    print(f"[DIAGNÓSTICO LASSO] O modelo eliminou {coeficientes_zerados} de {total_recursos} recursos da matriz.\n")
    
    return modelo

def executar_naive_bayes(X_train, X_test, y_train, y_test):
    print("\n[ALGORITMO] Inicializando Gaussian Naive Bayes OTIMIZADO...")
    
    print("Aplicando padronização de recursos (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # OTIMIZAÇÃO 1: Injetando probabilidades a priori customizadas para forçar ceticismo comprador
    # Forçamos o modelo a iniciar assumindo 60% de chance de Queda e 40% de Alta
    modelo = GaussianNB(priors=[0.60, 0.40])
    
    print("Treinando classificador probabilístico Bayesiano...")
    modelo.fit(X_train_scaled, y_train)
    print("Treinamento concluído!")
    
    # OTIMIZAÇÃO 2: Interceptando as probabilidades do Teorema de Bayes
    print("Aplicando Limiar de Decisão Ajustado (Threshold: 55%)...")
    probabilidades = modelo.predict_proba(X_test_scaled)
    
    # Só autoriza o sinal de Alta (1) se a probabilidade calculada for >= 55%
    limiar = 0.55
    y_pred = (probabilidades[:, 1] >= limiar).astype(int)
    
    exibir_metricas("NAIVE BAYES OTIMIZADO", y_test, y_pred)
    return modelo

def exibir_metricas(nome_modelo, y_real, y_previsto):
    acuracia = accuracy_score(y_real, y_previsto)
    matriz = confusion_matrix(y_real, y_previsto)
    
    print("\n==================================================")
    print(f"   RESULTADOS {nome_modelo} (Acurácia: {acuracia:.2%})")
    print("==================================================")
    print("\n--- Relatório de Classificação Detalhado ---")
    print(classification_report(y_real, y_previsto, target_names=['Queda (0)', 'Alta (1)']))
    print("--- Matriz de Confusão ---")
    print(f"                  Previsão Queda    Previsão Alta")
    print(f"Realidade Queda:       {matriz[0][0]}               {matriz[0][1]}")
    print(f"Realidade Alta:        {matriz[1][0]}               {matriz[1][1]}")
    print("==================================================")