# Detecção de Fraudes em Transações de Cartão de Crédito

Projeto de Parceria EBAC x Semantix — pipeline completo de machine learning para detecção de fraudes,
com foco em impacto financeiro real para o negócio.

**Resultado final:** saldo líquido de +USD 866.004 no conjunto de teste, com apenas 224 bloqueios
indevidos e Precision de 88%.

---

## O problema

O mercado de pagamentos digitais brasileiro movimenta trilhões de reais por ano. Empresas como
PicPay, BoaVista e Travelex lidam diariamente com o desafio de identificar transações fraudulentas
em tempo real — sem bloquear clientes legítimos e sem impactar a experiência do usuário.

Regras manuais não escalam. Um modelo que aprende padrões comportamentais diretamente dos dados é
fundamentalmente superior.

---

## Dataset

- **Fonte:** [Credit Card Transactions Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Licença:** ODbL (Open Database License)
- **Treino:** 1.296.675 transações
- **Teste:** 555.719 transações
- **Desbalanceamento:** 99,42% legítimas / 0,58% fraudes (razão 171:1)

---

## Pipeline

```
Dados brutos
    │
    ├── EDA (distribuição, valor, categoria, faixa etária)
    │
    ├── Pré-processamento
    │       ├── Remoção de colunas irrelevantes
    │       ├── Feature engineering (idade, hora, dia_semana, distancia)
    │       └── Label Encoding das categóricas
    │
    ├── Balanceamento
    │       ├── Estratégia 1: SMOTE + Undersampling
    │       └── Estratégia 2: scale_pos_weight (vencedor)
    │
    ├── Modelagem
    │       ├── Árvore de Decisão (baseline)
    │       ├── XGBoost + SMOTE
    │       └── XGBoost + scale_pos_weight
    │
    ├── Avaliação
    │       ├── Métricas (Precision, Recall, F1, AUC-ROC)
    │       ├── Matrizes de confusão
    │       ├── Curva Precision-Recall e análise de threshold
    │       └── Impacto financeiro (saldo líquido)
    │
    ├── Cross-Validation (Stratified K-Fold, 5 folds)
    │
    └── Otimização de Hiperparâmetros (RandomizedSearchCV)
```

---

## Resultados

### Comparação dos modelos

| Modelo | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| Árvore de Decisão | 0.16 | 0.93 | 0.27 | 0.9842 |
| XGBoost SMOTE | 0.19 | 0.90 | 0.32 | 0.9892 |
| XGBoost scale_pos_weight | 0.28 | 0.95 | 0.43 | 0.9972 |
| **XGBoost SPW otimizado** | **0.78** | **0.81** | **0.79** | **0.9931** |

### Impacto financeiro por cenário

| Cenário | Fraudes evitadas | Legítimas bloqueadas | Saldo líquido |
|---|---|---|---|
| SMOTE, threshold 0.5 | USD 1.122.095 | USD 2.050.334 | **-USD 928.239** |
| SMOTE, threshold 0.97 | USD 988.132 | USD 329.820 | **+USD 658.312** |
| SPW, threshold 0.5 | USD 1.122.442 | USD 1.810.154 | **-USD 687.712** |
| SPW, threshold 0.98 | USD 998.614 | USD 183.964 | **+USD 814.651** |
| **SPW otimizado, threshold 0.85** | **USD 1.009.028** | **USD 143.024** | **+USD 866.004** |

---

## Principais descobertas

- **Acurácia não é a métrica certa.** Com 99,42% de legítimas, qualquer modelo que classifica tudo
como legítimo teria 99,42% de Acurácia — e detectaria zero fraudes.

- **SMOTE gera saldo negativo com threshold padrão.** A síntese artificial de dados cria fronteiras
de decisão mais agressivas, bloqueando legítimas em excesso.

- **scale_pos_weight supera SMOTE em todas as métricas.** Treinar nos dados reais com penalização
assimétrica produz padrões mais precisos e menos falsos positivos.

- **O threshold é uma decisão de negócio.** O mesmo modelo com threshold 0.5 tem saldo -USD 688k;
com threshold 0.85 tem saldo +USD 866k. A diferença é de USD 1,55 milhão.

- **amt responde por 50% da importância do modelo.** Fraudadores concentram ataques em transações
de alto valor, especialmente nas categorias shopping_net e misc_net.

---

## Modelo final recomendado

**XGBoost com scale_pos_weight otimizado, threshold 0.85**

| Hiperparâmetro | Valor |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 8 |
| `learning_rate` | 0.3 |
| `subsample` | 0.8 |
| `colsample_bytree` | 1.0 |
| `min_child_weight` | 3 |
| `gamma` | 0.3 |
| `scale_pos_weight` | 171.8 |

---

## Tecnologias

- Python 3.10
- pandas, numpy
- scikit-learn
- imbalanced-learn
- XGBoost
- matplotlib, seaborn

---

## Estrutura do repositório

```
├── Semantix.ipynb      # Notebook principal
├── fraudTrain.csv         # Dataset de treino (download via Kaggle)
├── fraudTest.csv          # Dataset de teste (download via Kaggle)
└── README.md
```

---

## Como executar

```bash
# instalar dependências
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

# baixar o dataset
# acesse https://www.kaggle.com/datasets/kartik2112/fraud-detection
# faça o download e coloque fraudTrain.csv e fraudTest.csv na raiz do projeto

# executar o notebook
jupyter notebook Semantix_v4.ipynb
```

---

**Autor:** João Alfredo de Sousa Siqueira  
**Curso:** Cientista de Dados — EBAC  
**Parceria:** Semantix  
