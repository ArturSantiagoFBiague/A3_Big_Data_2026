from pipeline_dados import extrair_e_enriquecer_dados, preparar_matriz_treino_teste
import modelos

def executar_pipeline_completo(algoritmo_escolhido="random_forest"):
    # 1. Executa o pipeline ETL e de Engenharia de Recursos
    extrair_e_enriquecer_dados()
    
    # 2. Carrega, analisa e faz a separação temporal dos conjuntos
    X_train, X_test, y_train, y_test, pesos = preparar_matriz_treino_teste()
    
    # 3. Seletor do algoritmo de treino
    if algoritmo_escolhido == "random_forest":
        modelos.executar_random_forest(X_train, X_test, y_train, y_test, pesos)
        
    elif algoritmo_escolhido == "logistica_l1":
        modelos.executar_logistica_l1(X_train, X_test, y_train, y_test, pesos)
        
    elif algoritmo_escolhido == "naive_bayes":
        modelos.executar_naive_bayes(X_train, X_test, y_train, y_test)
        
    else:
        print(f"Erro: Algoritmo '{algoritmo_escolhido}' não reconhecido.")

if __name__ == "__main__":
    # Altere a string abaixo para escolher o algoritmo desejado:
    # Opções atuais: "random_forest", "logistica_l1", "naive_bayes"
    ALGORITMO = "logistica_l1"
    
    executar_pipeline_completo(algoritmo_escolhido=ALGORITMO)