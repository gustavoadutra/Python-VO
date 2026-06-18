#!/bin/bash

# Define a pasta onde os arquivos estao
CONFIG_DIR="params"

# Verifica se a pasta existe
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Erro: A pasta $CONFIG_DIR não foi encontrada."
    exit 1
fi

# Loop pelos arquivos .yaml na pasta
for config_file in "$CONFIG_DIR"/*.yaml; do
    echo "----------------------------------------------------"
    echo "Executando: python main.py --config $config_file"
    echo "----------------------------------------------------"
    
    # Executa o comando
    python main.py --config "$config_file"
    
    # Opcional: Verifica se houve erro antes de continuar
    if [ $? -ne 0 ]; then
        echo "Erro ao executar $config_file. Encerrando."
        exit 1
    fi
done

echo "Todas as execuções foram concluídas."