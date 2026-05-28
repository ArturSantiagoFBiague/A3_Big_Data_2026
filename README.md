### Preparar Ambiente de Desenvolvimento 

Digite os seguintes comandos no *Terminal*

python3 -m venv .venv
source .venv/bin/activate

pip install -r requeriments.txt


### Rodar Projeto
De o comando:

python3 main.py

Lembrando que podemos alternar em Hardcode entre 3 tipos diferentes de *Algoritimos*

Modifique final do arquivo main.py para:

## Algoritimo Randon Forest
if __name__ == "__main__":
    
    ALGORITMO = "random_forest"
    executar_pipeline_completo(algoritmo_escolhido=ALGORITMO)

## Algoritimo Regressão Logistica L1
if __name__ == "__main__":
    
    ALGORITMO = "logistica_l1"
    executar_pipeline_completo(algoritmo_escolhido=ALGORITMO)


## Algoritimo Naive Bayes 

if __name__ == "__main__":
    
    ALGORITMO = "naive_bayes"
    executar_pipeline_completo(algoritmo_escolhido=ALGORITMO)