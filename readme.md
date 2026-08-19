# Modelo para Classificação de Batimentos Ectópicos - CNN

Este projeto consiste no desenvolvimento de uma **Rede Neural Convolucional (CNN)** para a classificação de batimentos cardíacos ectópicos a partir de sinais de eletrocardiograma (ECG).

O objetivo é avaliar o desempenho da CNN em dois cenários:

- **Classificação intra-paciente:** treinamento e avaliação utilizando dados provenientes dos mesmos pacientes.
- **Classificação inter-paciente:** treinamento e avaliação utilizando pacientes diferentes, permitindo analisar a capacidade de generalização do modelo.

A construção do modelo foi baseada em trabalhos da literatura relacionados à utilização de Redes Neurais Convolucionais para classificação de sinais de ECG.

## 📚 Artigos de referência

Os principais trabalhos utilizados como referência foram:

- **ECG Heartbeat Classification Using Convolutional Neural Networks**  
  _XueXiang Xu e HongXing Liu._

- **Real-Time Patient-Specific ECG Classification by 1-D Convolutional Neural Networks**  
  _Serkan Kiranyaz, Turker Ince e Moncef Gabbouj._

Esses trabalhos serviram como base para a definição da arquitetura da CNN e da abordagem utilizada na classificação dos batimentos.

## 🫀 Base de dados

Foi utilizada a **MIT-BIH Arrhythmia Database**, uma base amplamente utilizada em estudos relacionados à análise e classificação de sinais de ECG.

A base contém **48 registros de pacientes**, cada um contendo sinais de eletrocardiograma.

### Segmentação dos batimentos

Os sinais foram segmentados a partir dos **picos R** detectados no ECG. Cada batimento foi centralizado no respectivo pico R e representado por um segmento contendo **180 amostras**.

O processo pode ser representado da seguinte forma:

```text
Sinal ECG
   │
   ▼
Detecção dos picos R
   │
   ▼
Localização dos batimentos
   │
   ▼
Segmentação centrada no pico R
   │
   ▼
180 amostras por batimento
   │
   ▼
Entrada da CNN
```

## Arquitetura CNN

A arquitetura utilizada é baseada em uma CNN 1-D, desenvolvida para trabalhar diretamente com os segmentos dos sinais de ECG.

A rede é composta por:

4 camadas convolucionais;
2 camadas de Max Pooling;
Camadas de Dropout para regularização;
3 camadas totalmente conectadas (Fully Connected);
Camada de saída para classificação em duas classes.

A arquitetura pode ser resumida como:

As camadas de Dropout foram utilizadas como estratégia de regularização, buscando reduzir o risco de overfitting durante o treinamento.

## 📊 Avaliação

O modelo será avaliado considerando dois cenários:

### Intra-paciente

No cenário intra-paciente, os dados utilizados no treinamento, validação e teste pertencem aos mesmos pacientes.

Essa abordagem permite avaliar o desempenho do modelo quando ele já teve contato com características específicas dos pacientes durante o treinamento.

### Inter-paciente

No cenário inter-paciente, os pacientes utilizados no treinamento são diferentes daqueles utilizados na validação e no teste.

Essa abordagem é utilizada para avaliar a capacidade de generalização da CNN para pacientes que não participaram do treinamento.

### Avaliação

O desempenho da CNN será analisado utilizando métricas de classificação e ferramentas de avaliação, como:

- Acurácia;
- Matriz de confusão;
- Precisão;
- Sensibilidade;
- Especificidade;
- F1-Score.

Além das métricas, os resultados dos cenários intra-paciente e inter-paciente serão comparados para avaliar o impacto da estratégia de divisão dos dados no desempenho do modelo.

### 🚧 Status do projeto

Em desenvolvimento
