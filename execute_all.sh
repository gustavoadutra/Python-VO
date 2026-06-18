#!/bin/bash

# Define a pasta dos arquivos
CONFIG_DIR="params"
DATASET=$1
PNP_FLAG=$2
BA_FLAG=$3

# Verifica se o dataset foi informado
if [ -z "$DATASET" ]; then
    echo "Uso: $0 [kaist|cusco|kitti] [--no-pnp] [--ba]"
    exit 1
fi

# Validação do dataset
DATASET=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
if [[ "$DATASET" != "kaist" && "$DATASET" != "cusco" && "$DATASET" != "kitti" ]]; then
    echo "Erro: Dataset inválido. Escolha entre: kaist, cusco ou kitti."
    exit 1
fi

# Montagem dos argumentos dinâmicos
ARGS=""

# Verifica se --no-pnp foi passado (pode ser no 2º ou 3º argumento)
if [[ "$PNP_FLAG" == "--no-pnp" || "$BA_FLAG" == "--no-pnp" ]]; then
    ARGS="$ARGS --no-pnp"
fi

# Verifica se --ba foi passado (pode ser no 2º ou 3º argumento)
if [[ "$PNP_FLAG" == "--ba" || "$BA_FLAG" == "--ba" ]]; then
    ARGS="$ARGS --ba"
fi

echo "Iniciando processamento para: $DATASET"
echo "Argumentos extras: $ARGS"

# Loop pelos arquivos do dataset escolhido
for config_file in "$CONFIG_DIR"/"$DATASET"_*.yaml; do
    echo "----------------------------------------------------"
    echo "Executando: python main.py --config $config_file $ARGS"
    echo "----------------------------------------------------"
    
    python main.py --config "$config_file" $ARGS
done

echo "Concluído: Todos os arquivos de $DATASET foram processados."