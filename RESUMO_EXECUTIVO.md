# 📊 Resumo Executivo - Trabalho 2 RNP

## 📚 Navegação

- **[📖 README.md](README.md)** - Documentação completa
- **[🚀 COMO_USAR.md](COMO_USAR.md)** - Guia rápido de uso
- **[📊 RESUMO_EXECUTIVO.md](RESUMO_EXECUTIVO.md)** - Visão geral do projeto (você está aqui)
- **[📈 RESULTADOS_FINAIS.md](RESULTADOS_FINAIS.md)** - Resultados dos experimentos

---

## 🎯 Objetivo
Implementar e comparar três arquiteturas de redes neurais (MLP, LSTM, GRU) para predição da série temporal Mackey-Glass, com múltiplas configurações e variações para análise abrangente.

## ✅ Status do Projeto
**IMPLEMENTAÇÃO COMPLETA** ✅

## 🏗️ Arquitetura Implementada

### Modelos Disponíveis (15 configurações)

> **📖 Para detalhes completos, consulte [README.md](README.md#modelos-disponíveis)**

#### MLP (Multi-Layer Perceptron)
- **mlp_small**: 2 camadas [64, 32], ~3K parâmetros
- **mlp_medium**: 3 camadas [128, 64, 32], ~14K parâmetros  
- **mlp_large**: 4 camadas [256, 128, 64, 32], ~59K parâmetros

#### LSTM (Long Short-Term Memory)
- **lstm_small**: 1 camada, 32 unidades, ~8K parâmetros
- **lstm_medium**: 2 camadas, 64 unidades, ~50K parâmetros
- **lstm_large**: 3 camadas, 128 unidades, ~200K parâmetros
- **lstm_bidirectional**: 2 camadas bidirecionais, 64 unidades
- **lstm_attention**: 2 camadas com atenção, 64 unidades

#### GRU (Gated Recurrent Unit)
- **gru_small**: 1 camada, 32 unidades, ~6K parâmetros
- **gru_medium**: 2 camadas, 64 unidades, ~37K parâmetros
- **gru_large**: 3 camadas, 128 unidades, ~150K parâmetros
- **gru_bidirectional**: 2 camadas bidirecionais, 64 unidades
- **gru_attention**: 2 camadas com atenção, 64 unidades

## 🚀 Como Executar

> **🚀 Para instruções detalhadas, consulte [COMO_USAR.md](COMO_USAR.md)**

### Experimento Padrão (Modelos Principais)
```bash
cd mackey_glass_prediction/experiments
python run_experiment.py
```
**Executa**: `mlp_large`, `lstm_large`, `gru_large`

### Experimento Completo (Todos os Modelos)
```bash
python run_experiment.py --models all
```
**Executa**: Todos os 15 modelos (pode demorar várias horas)

### Teste Rápido
```bash
python run_experiment.py --models mlp_small lstm_small gru_small --no-save
```

## 📁 Estrutura do Projeto

```
Trabalho2RNP/
├── RESULTADOS_FINAIS.md              # Relatório pós-experimentos
├── COMO_USAR.md                      # Guia detalhado de uso
├── RESUMO_EXECUTIVO.md               # Este arquivo
└── mackey_glass_prediction/          # Código fonte
    ├── config/config.py              # 15 configurações de modelos
    ├── data/mackey_glass_generator.py # Geração da série temporal
    ├── models/                       # 3 tipos de modelos
    │   ├── mlp_model.py             # MLP com variações
    │   ├── lstm_model.py            # LSTM + bidirectional + attention
    │   └── gru_model.py             # GRU + bidirectional + attention
    ├── utils/                        # Módulos auxiliares
    │   ├── training.py              # Treinamento e avaliação
    │   └── visualization/           # 7 módulos de visualização
    ├── experiments/                  # Scripts de execução
    │   └── run_experiment.py        # Script principal
    ├── generate_interactive_report.py # Relatório HTML demonstrativo
    └── requirements.txt             # 22 dependências
```

## ⚙️ Configurações do Experimento

### Série Mackey-Glass
- **Pontos**: 10.000
- **Parâmetros**: τ=20, β=0.4, γ=0.2, n=18, x₀=0.8
- **Janela de entrada**: 20 pontos
- **Predição**: 1 passo à frente

### Dataset
- **Divisão**: 90% treino, 10% validação
- **Batch size**: 8192
- **Normalização**: Min-Max Scaling

### Treinamento
- **Épocas máximas**: 150
- **Early stopping**: 15 épocas de paciência
- **Otimizador**: Adam (lr=1e-3)
- **Scheduler**: ReduceLROnPlateau
- **Regularização**: L2 (1e-5) + Dropout

## 📊 Saídas Geradas

### Estrutura de Resultados
```
experiments/results/final_report_[timestamp]/
├── 01_visao_geral_[timestamp].png           # Comparação geral
├── [modelo]/                               # Para cada modelo:
│   ├── training_curves.png                 # Curvas de loss
│   ├── predictions.png                     # Predições vs reais
│   ├── [modelo]_qq_plot_[timestamp].png    # Análise Q-Q
│   ├── [modelo]_cdf_[timestamp].png        # Análise CDF
│   ├── best_model.pth                      # Melhor modelo
│   └── final_model.pth                     # Modelo final
├── 99_tabela_metricas_[timestamp].png      # Tabela formatada
├── 99_comparacao_metricas_[timestamp].png  # Comparação visual
├── metrics_table.csv                       # Dados em CSV
└── relatorio.html                          # Relatório interativo
```

### Métricas Avaliadas

#### Métricas Básicas
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coeficiente de determinação

#### Métricas Normalizadas
- **EQMN1**: Erro Quadrático Médio Normalizado pela Variância
  - **Fórmula**: EQMN1 = MSE / Var(y_true)
  - **Interpretação**: < 0.1 excelente, < 0.5 bom
- **EQMN2**: Erro Quadrático Médio Normalizado pelo Modelo Naive
  - **Fórmula**: EQMN2 = MSE / MSE_naive (modelo de persistência)
  - **Interpretação**: < 1.0 melhor que naive, < 0.5 excelente

## ✨ Funcionalidades Implementadas

### 🧠 Modelos Avançados
- ✅ MLP com múltiplas camadas e dropout adaptativo
- ✅ LSTM com opções bidirecionais e mecanismo de atenção
- ✅ GRU com opções bidirecionais e mecanismo de atenção
- ✅ Arquiteturas escaláveis (small, medium, large)

### 🔬 Treinamento Robusto
- ✅ Early stopping inteligente
- ✅ Learning rate scheduling
- ✅ Regularização L2 + Dropout
- ✅ Checkpoint automático (melhor modelo)
- ✅ Seeds fixas para reprodutibilidade

### 📊 Visualizações Completas
- ✅ Curvas de treinamento e validação
- ✅ Comparação de predições vs valores reais
- ✅ Análise estatística de resíduos (Q-Q plots, CDF)
- ✅ Tabelas de métricas formatadas
- ✅ Gráficos comparativos entre modelos
- ✅ Relatórios HTML interativos

### 🛠️ Infraestrutura
- ✅ Configurações centralizadas
- ✅ Modularidade e extensibilidade
- ✅ Interface de linha de comando flexível
- ✅ Detecção automática GPU/CPU
- ✅ Logging detalhado
- ✅ Tratamento de erros

### 📈 Análises Estatísticas
- ✅ Distribuição de resíduos
- ✅ Testes de normalidade
- ✅ Análise de autocorrelação
- ✅ Métricas múltiplas (R², MAPE, EQMN1, EQMN2, etc.)
- ✅ Comparação estatística entre modelos

## 🎛️ Opções de Execução

### Básicas
```bash
python run_experiment.py                    # Modelos principais
python run_experiment.py --models all       # Todos os modelos
python run_experiment.py --no-save          # Sem salvar resultados
```

### Específicas
```bash
python run_experiment.py --models mlp_medium lstm_attention
python run_experiment.py --output-dir meu_experimento
python run_experiment.py --models gru_large --no-save
```

### Análise
```bash
python generate_interactive_report.py       # Relatório demonstrativo
```

## 🏆 Características Técnicas

### Escalabilidade
- 3 tipos de modelos × 5 configurações = 15 modelos
- Parâmetros de 3K (mlp_small) até 200K (lstm_large)
- Suporte para experimentação em larga escala

### Robustez
- Tratamento automático de GPU/CPU
- Early stopping para evitar overfitting
- Validação cruzada temporal
- Normalização e desnormalização automática

### Reprodutibilidade
- Seeds fixas para todos os experimentos
- Configurações versionadas
- Checkpoints automáticos
- Logs detalhados de execução

### Usabilidade
- Interface simples e intuitiva
- Documentação abrangente
- Relatórios visuais automáticos
- Debugging facilitado com modo --no-save

## 📋 Próximos Passos

1. **Executar experimentos**: `python run_experiment.py`
2. **Analisar resultados**: Verificar pasta `results/final_report_*/`
3. **Comparar modelos**: Consultar `metrics_table.csv`
4. **Gerar relatório**: `python generate_interactive_report.py`
5. **Atualizar RESULTADOS_FINAIS.md**: Com métricas reais obtidas

## 📚 Links Úteis

- **[🚀 COMO_USAR.md](COMO_USAR.md)** - Guia prático de execução
- **[📖 README.md](README.md)** - Documentação técnica completa
- **[📈 RESULTADOS_FINAIS.md](RESULTADOS_FINAIS.md)** - Análise dos resultados experimentais

---
**Desenvolvido por**: Rafael Ratacheski de Sousa Raulino  
**Mestrando em Engenharia Elétrica e de Computação - UFG**  
**Disciplina**: Redes Neurais Profundas - 2025/1  
**Data**: 29 de Maio de 2025  
**Status**: ✅ IMPLEMENTAÇÃO COMPLETA