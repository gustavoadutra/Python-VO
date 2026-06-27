#!/bin/bash

# Define a pasta dos arquivos
CONFIG_DIR="params"

# Verifica se o dataset foi informado no primeiro argumento
if [ -z "$1" ]; then
    echo "Uso: $0 <dataset> [--no-pnp] [--ba] [--max-frames <numero>]"
    echo "Exemplo: $0 kitti --ba --max-frames 100"
    exit 1
fi

# Validação do dataset
DATASET=$(echo "$1" | tr '[:upper:]' '[:lower:]')
if [[ "$DATASET" != "kaist" && "$DATASET" != "cusco" && "$DATASET" != "kitti" ]]; then
    echo "Erro: Dataset inválido. Escolha entre: kaist, cusco ou kitti."
    exit 1
fi

# Remove o primeiro argumento (dataset) da fila para processar os opcionais
shift 

# Montagem dos argumentos dinâmicos
ARGS=""

# Loop para ler todos os argumentos extras em qualquer ordem
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-pnp)
            ARGS="$ARGS --no-pnp"
            shift # Pula pro próximo argumento
            ;;
        --ba)
            ARGS="$ARGS --ba"
            shift # Pula pro próximo argumento
            ;;
        --max-frames)
            # Verifica se o próximo argumento existe e não é outra flag (começando com -)
            if [ -n "$2" ] && [[ ${2:0:1} != "-" ]]; then
                ARGS="$ARGS --max-frames $2"
                shift 2 # Pula a flag e o número passado
            else
                echo "Erro: Você precisa informar um número após --max-frames."
                exit 1
            fi
            ;;
        *)
            echo "Aviso: Argumento desconhecido ignorado: $1"
            shift
            ;;
    esac
done

echo "Iniciando processamento para: $DATASET"
echo "Argumentos extras: $ARGS"

# Loop pelos arquivos do dataset escolhido
for config_file in "$CONFIG_DIR"/"$DATASET"_*.yaml; do
    # Previne que o script tente rodar literalmente "*.yaml" se não houver arquivos
    if [ ! -e "$config_file" ]; then
        echo "Nenhum arquivo de configuração encontrado para o padrão ${DATASET}_*.yaml na pasta ${CONFIG_DIR}."
        break
    fi

    echo "----------------------------------------------------"
    echo "Executando: python main.py --config $config_file $ARGS"
    echo "----------------------------------------------------"
    
    python main.py --config "$config_file" $ARGS
done

echo "Concluído: Todos os arquivos de $DATASET foram processados."