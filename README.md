"""# Predição de Severidade da Dor Crônica Neuropática via EEG de Repouso

Este repositório contém o pipeline completo de processamento de sinais e Machine Learning para a identificação de biomarcadores de dor crônica neuropática a partir de Eletroencefalograma (EEG) em estado de repouso.

O objetivo central foi cruzar dados clínicos de questionários de dor (**BPI** e **PainDetect**) com sinais elétricos cerebrais ($N=36$ pacientes), buscando uma assinatura neurofisiológica objetiva para a intensidade da dor.

---

## 1. Pré-Processamento dos Sinais

O pré-processamento foi realizado utilizando a biblioteca `MNE-Python`:

* **Importação e Estruturação:** Leitura de arquivos `.fif` (e `.gdf`) padronizados.
* **Filtragem Espacial e Temporal:** Aplicação de filtros *High-pass* (4 Hz) e *Low-pass* (40 Hz) para isolar as frequências cognitivas (Theta, Alpha, Beta e Gamma) e remover ruídos.
* **Alinhamento de Dados:** Criação de um pipeline robusto com `pandas` para cruzar o ID do paciente no arquivo de EEG com sua respectiva nota de dor nos questionários clínicos, utilizando estratégias contra valores ausentes e tipagem incorreta (*Zero Padding* e *casting* seguro).

---

## 2. Hipóteses Testadas e Modelos que "Falharam"

Testamos abordagens clássicas e avançadas que esbarraram nas limitações físicas do tamanho da amostra:

### Regressão Contínua (SVR e Random Forest Regressor)
* **Objetivo:** Tentar prever a nota exata de dor (de 0 a 10) informada no questionário BPI.
* **O que aconteceu:** *Overfitting* severo ($R^2$ negativo e altos erros RMSE).
* **Conclusão:** Com métodos lineares e apenas 36 pacientes o modelo decorava os ruídos do treino e falhava na validação cruzada.

### Acoplamento Fase-Amplitude - PAC
* **Objetivo:** Medir como ondas lentas modulam ondas rápidas (ex: modulação Alpha-Gamma na região frontal).
* **O que aconteceu:** O PAC gerou 248 variáveis (biomarcadores) por paciente. Com 248 *features* e apenas 36 amostras, a acurácia do modelo de Classificação despencou para ~39%.
* **Conclusão:** O algoritmo Random Forest "se afogou" na dimensionalidade, enquanto o `SelectKBest` sofreu ilusão estatística durante a Validação Cruzada K-Fold, selecionando falsos padrões gerados pelo ruído.

---

## 3. Decomposição EMD + Classificação

Para contornar o baixo número de amostras, mudamos o paradigma:

1.  **Problema Binário:** Abandonamos a regressão e passamos a classificar os pacientes em dois grupos clínicos reais: **Dor Intensa ($\ge 6$)** vs. **Dor Leve/Moderada ($< 6$)**.
2.  **Decomposição em Modos Empíricos (EMD):** O cérebro não é linear, então ao em vez de usar Fourier (PSD), utilizamos o EMD (via pacote `EMD-signal`) para "descascar" o sinal bruto em Funções de Modo Intrínseco (IMFs) apenas nos canais de ouro (**C3, C4, P7, P8, F3, F4**) que foram evidenciados pelos resultados dos modelos anteriores.

### Resultados do Machine Learning
Utilizando um **Random Forest Classifier** com Validação Cruzada Estratificada (5 Folds), obtivemos:
* **Acurácia Média:** ~61.1%
* **Feature Importances:** O modelo focou quase 80% de sua decisão em apenas duas regiões do hemisfério direito:
    * `EMD_C4_IMF1_Entropia`: O nível de caos/desorganização nas ondas rápidas do Córtex Somatossensorial (C4).
    * `EMD_P8_IMF2_Energia`: A força/variância das ondas médias no Lobo Parietal (P8).

---

### Arquivos Principais

* **`preprocessamento.ipynb`**: Script de limpeza de dados MNE e geração da matriz mesclada com questionários.
* **`modelos_regressao_pac.ipynb`**: Registro dos testes de baseline (SVM Regressor) e extração de PAC (Phase-Amplitude Coupling).
* **`pipeline_emd_classificacao.ipynb`**: Pipeline final contendo a extração EMD, o modelo de Classificação Random Forest e a geração dos gráficos estatísticos.
* **`tabelas/`**: Diretório esperado para os arquivos CSV (Demographics, BPI, PainDetect).
* **`eeg_data/`**: Diretório esperado para os arquivos .fif pré-processados.
