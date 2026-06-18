#!/bin/bash

# Define a pasta dos arquivos
CONFIG_DIR="params"
DATASET=$1

# Verifica se o usuário informou um dataset
if [ -z "$DATASET" ]; then
    echo "Uso: $0 [kaist|cusco|kitti]"
    exit 1
fi

# Converte o argumento para letras minúsculas para evitar erros de digitação
DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

# Verifica se o dataset informado é válido
if [[ "$DATASET" != "kaist" && "$DATASET" != "cusco" && "$DATASET" != "kitti" ]]; then
    echo "Erro: Dataset inválido. Escolha entre: kaist, cusco ou kitti."
    exit 1
fi

echo "Iniciando processamento para: $DATASET"

# Executa apenas os arquivos que contêm o nome do dataset
for config_file in "$CONFIG_DIR"/"$DATASET"_*.yaml; do
    echo "----------------------------------------------------"
    echo "Executando: python main.py --config $config_file"
    echo "----------------------------------------------------"
    
    python main.py --config "$config_file"
done

echo "Concluído: Todos os arquivos de $DATASET foram processados."