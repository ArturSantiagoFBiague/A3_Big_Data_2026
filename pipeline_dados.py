import yfinance as yf
import pandas as pd
import numpy as np

def extrair_e_enriquecer_dados(start_date="2025-01-01", end_date="2026-05-01", arquivo_saida="bitcoin_matriz_recursos_final_1h.csv"):
    print("--- ETAPA 1: Baixando dados horários (1h) do Yahoo Finance... ---")
    dados_brutos = yf.download("BTC-USD", start=start_date, end=end_date, interval="1h")
    dados_brutos.to_csv("raw_data_1h.csv")

    df = dados_brutos.copy()
    df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    print("\n--- ETAPA 2: Calculando métricas complementares horárias... ---")
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['Volatilidade_24h'] = df['Close'].rolling(window=24).std()
    df['Vol_Media_24h'] = df['Volume'].rolling(window=24).mean()

    delta = df['Close'].diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = ganho / perda
    df['RSI_14'] = 100 - (100 / (1 + rs))

    print("\n--- ETAPA 3: Estruturando Janela Deslizante (24h) e Alvo Futuro (24h)... ---")
    df['Preco_Futuro'] = df['Close'].shift(-24)
    df['Target'] = np.where(df['Preco_Futuro'] > df['Close'], 1, 0)

    features_para_atrasar = ['Close', 'RSI_14', 'Volatilidade_24h']
    for coluna in features_para_atrasar:
        for lag in range(1, 25):
            df[f'{coluna}_Lag_{lag}'] = df[coluna].shift(lag)

    df_final = df.dropna().copy()
    df_final.index.name = 'Datetime'
    df_final.to_csv(arquivo_saida)
    print(f"\nSucesso! Matriz salva. Total de registros válidos: {len(df_final)} linhas.")
    return df_final

def preparar_matriz_treino_teste(caminho_csv="bitcoin_matriz_recursos_final_1h.csv"):
    df = pd.read_csv(caminho_csv, index_col='Datetime', parse_dates=True)
    
    print("\n--- ANÁLISE DE BALANCEAMENTO DOS DADOS ---")
    total_registros = len(df)
    contagem_classes = df['Target'].value_counts()
    proporcoes = df['Target'].value_counts(normalize=True) * 100

    print(f"Total de horas catalogadas: {total_registros}")
    for classe, qtd in contagem_classes.items():
        label = "Alta (1)" if classe == 1 else "Queda (0)"
        print(f"  Classe {label}: {qtd} horas ({proporcoes[classe]:.2f}%)")

    print("\n--- EXECUTANDO SEPARAÇÃO CRONOLÓGICA (80% Treino / 20% Teste) ---")
    ponto_divisao = int(total_registros * 0.80)

    colunas_preditivas = [col for col in df.columns if col not in ['Target', 'Preco_Futuro']]
    X = df[colunas_preditivas]
    y = df['Target']

    X_train, X_test = X.iloc[:ponto_divisao], X.iloc[ponto_divisao:]
    y_train, y_test = y.iloc[:ponto_divisao], y.iloc[ponto_divisao:]

    contagem_treino = y_train.value_counts()
    peso_classe_0 = total_registros / (2 * contagem_treino[0])
    peso_classe_1 = total_registros / (2 * contagem_treino[1])
    pesos_calculados = {0: peso_classe_0, 1: peso_classe_1}
    print(f"Pesos equilibrados gerados: {pesos_calculados}")

    return X_train, X_test, y_train, y_test, pesos_calculados