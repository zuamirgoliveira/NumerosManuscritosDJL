# Relatório final — decisões e resultados

Este relatório reúne as decisões tomadas no projeto e, ao final, explica os dados de [treinos.csv](treinos.csv) evolução por evolução. Os relatórios parciais continuam em [relatorio-baseline.md](relatorio-baseline.md), [relatorio-comparativo-mlp-cnn.md](relatorio-comparativo-mlp-cnn.md) e [relatorio-desempenho-carregamento.md](relatorio-desempenho-carregamento.md).

**Condições de todas as medições:** MNIST completo (60.000 imagens de treino, 10.000 de validação), 5 épocas, semente 42, lote 32 (salvo indicação), GPU NVIDIA RTX 5050 Laptop com 8 GB, CPU i7-14650HX com 24 threads e 14 GB de RAM.

---

## Parte 1 — Decisões

### D1. Semente fixa e registro de cada treino em CSV
- **Situação inicial:** sem semente, a mesma configuração dava 97,5%, 97,8% e 97,7% em execuções diferentes (variação de cerca de 0,3 ponto). Nessas condições não dá para saber se uma mudança melhorou o modelo.
- **Decisão:** semente padrão 42 (pesos iniciais e embaralhamento), definida antes de criar modelo e datasets. Cada treino acrescenta uma linha a `results/treinos.csv`.
- **Resultado:** duas execuções do MLP deram resultados idênticos. A partir daqui, uma diferença de acurácia é efeito da mudança feita, não do acaso.
- **Limite:** na CNN em GPU a semente não basta (ver D2).

### D2. CNN no lugar do MLP, mantendo os dois
- **Situação inicial:** o MLP achata a imagem em 784 números e não sabe quais pixels são vizinhos.
- **Decisão:** adicionar uma CNN (duas convoluções 3×3 com 32 e 64 filtros, max pooling 2×2, camada densa de 128) e manter o MLP como opção.
- **Resultado:** a acurácia de validação subiu de 97,65% para cerca de 98,6–98,9% na GPU, e os erros caíram de 235 para 113–136 em 10.000 imagens. O custo foi cerca de 20% mais tempo na GPU.
- **Limites:** a CNN praticamente exige GPU (6 a 7 vezes mais lenta em CPU), e o ganho foi medido só no MNIST.

### D3. Uma arquitetura por nome, cada uma com seu arquivo de pesos
- **Situação inicial:** com um único nome de modelo, treinar a CNN sobrescreveria o MLP.
- **Decisão:** um `enum Architecture` define a rede e o nome do arquivo (`mnist-mlp-…`, `mnist-cnn-…`). A escolha é feita por `-Dmodelo.arquitetura`.
- **Resultado:** os dois modelos coexistem e podem ser comparados. A estrutura da rede fica definida num único lugar, usada por treino e inferência. O padrão continua `mlp`, então CLI e GUI não mudaram de comportamento.

### D4. Cache em memória e montagem de lotes em threads
- **Situação inicial:** o MLP é tão pequeno que o tempo ia na leitura e decodificação dos PNGs. Por padrão o DJL monta os lotes na própria thread do treino, e a GPU ficava quase ociosa.
- **Decisão:** `-Dtreino.cache=true` (decodifica as 70.000 imagens uma vez, cerca de 55 MB) e `-Dtreino.threads=N` (monta os lotes em N threads daemon).
- **Resultado:** o treino ficou cerca de 2 vezes mais rápido, com a mesma acurácia (detalhes na Parte 2).
- **Padrão:** continuam desligados, para preservar o comportamento original.
- **Defeito encontrado:** com um pool de threads comum, o treino terminava e gravava o resultado, mas o processo nunca encerrava. O pool passou a usar threads daemon.

### D5. Escolha explícita do dispositivo e log do que está sendo usado
- **Situação inicial:** o DJL 0.36 não detectou a GPU (`GPUs disponíveis: 0`) porque não há CUDA Toolkit instalado, e o treino rodava em CPU sem avisar.
- **Decisão:** `-Dtreino.device=gpu|cpu|auto` e um log no início com engine, número de GPUs e dispositivo.
- **Resultado:** fica claro onde o treino roda, e é possível comparar CPU e GPU nas mesmas condições.

### D6. Perfil Maven `-Pgpu`
- **Situação inicial:** usar a GPU exigia lembrar de `PYTORCH_FLAVOR=cu128` e do dispositivo. A RTX 5050 (arquitetura Blackwell) precisa do build CUDA 12.8 ou mais novo.
- **Decisão:** um profile `gpu` no `pom.xml` configura as duas propriedades.
- **Resultado:** o procedimento fica versionado: `mvn exec:java -Pgpu ...`. Uma variável de ambiente `PYTORCH_FLAVOR`, mesmo vazia, tem prioridade sobre o perfil. Isso causou um erro durante os testes.

### D7. Avaliação em lote com matriz de confusão (`EvaluateMnist`)
- **Situação inicial:** CLI e GUI classificam uma imagem por vez, e a acurácia geral esconde onde o modelo erra.
- **Decisão:** uma classe que avalia uma pasta inteira e imprime acurácia geral, por dígito e matriz de confusão.
- **Resultado:** mede o modelo pelo mesmo caminho de inferência usado fora do treino. A CNN obteve 98,80% (9.880 de 10.000). O dígito 9 é o mais fraco (96,33%), confundido principalmente com 5 (11 vezes) e 4 (7 vezes).

### D8. Testes automatizados
- **Situação inicial:** não havia testes, e cada refatoração era verificada rodando o treino inteiro.
- **Decisão:** 13 testes JUnit 5 (`mvn test`), que usam PNGs sintéticos e os modelos versionados, sem precisar do MNIST nem de GPU.
- **Resultado:** o teste "cache igual ao disco" garante que a otimização de D4 não altera os dados. Outros garantem a saída 1×10 das duas redes e que as probabilidades somam 1.

### D9. `Model` fechado mesmo em caso de erro
O `Model` guarda memória nativa e só era fechado no fluxo normal. Passou a ser aberto em `try-with-resources`, como o resto do projeto.

### D10. Higiene do repositório
`images-training/`, `images-validation/` (70.000 PNGs) e `target/` ficam fora do git. Modelos treinados e relatórios ficam versionados. Os dados são regerados por um comando, e o `git status` fica limpo.

---

## Parte 2 — Resultados: o que o `treinos.csv` mostra

### Como ler as colunas

| Coluna | Significado |
|---|---|
| `acc_treino`, `acc_validacao` | Fração de acertos nas imagens de treino e nas 10.000 de validação, medida na última época |
| `perda_treino`, `perda_validacao` | Perda *softmax cross-entropy*; quanto menor, melhor |
| `tempo_total_s` | Tempo do `EasyTrain.fit` (treino e validação das 5 épocas), sem iniciar a JVM nem baixar o PyTorch |
| `tempo_epoca_s` | `tempo_total_s` dividido por 5 |
| `imagens_por_s` | 60.000 × 5 ÷ `tempo_total_s`; conta só as imagens de treino |

Quando a validação é 0,9765, o número de erros é (1 − 0,9765) × 10.000 = 235. Uso "erros" nessa escala nas tabelas abaixo.

### Evolução 0 — Baseline (MLP, GPU, sem otimizações)

| Linha | Acc. treino | Acc. validação | Erros | Perda treino | Perda validação | Tempo | Imagens/s |
|---|---|---|---|---|---|---|---|
| baseline-mlp-r1 | 0,9868 | 0,9765 | 235 | 0,0409 | 0,0802 | 24,8 s | 12.110 |
| baseline-mlp-r2 | 0,9868 | 0,9765 | 235 | 0,0409 | 0,0802 | 23,1 s | 12.960 |

- **Reprodutibilidade (D1):** as colunas de acurácia e perda são idênticas nas duas linhas. Só o tempo varia (7%), por causa do estado da máquina e não do treino. Esse é o efeito da semente fixa.
- **Sobreajuste leve:** a acurácia de treino é 1,03 ponto maior que a de validação, e a perda de validação é 1,96 vez a de treino. A rede já se ajusta um pouco ao treino.
- **Tempo:** 12.000 a 13.000 imagens por segundo é pouco para uma GPU com um modelo tão pequeno. Isso aponta para o gargalo de dados, tratado na evolução 3.

### Evolução 1 — CNN em GPU

| Linha | Acc. treino | Acc. validação | Erros | Perda treino | Perda validação | Tempo | Imagens/s |
|---|---|---|---|---|---|---|---|
| cnn-r1 | 0,9964 | 0,9877 | 123 | 0,0104 | 0,0490 | 28,7 s | 10.452 |
| cnn-r2 | 0,9967 | 0,9864 | 136 | 0,0098 | 0,0588 | 28,9 s | 10.367 |

- **Ganho de acurácia:** a validação passa de 0,9765 para 0,9864–0,9877. Os erros caem de 235 para 123–136 (cerca de 45% a menos). A acurácia de treino sobe de 98,7% para 99,6%, ou seja, a CNN aprende mais padrões.
- **Custo:** o tempo vai de ~24 s para ~29 s (cerca de 20% a mais), e as imagens por segundo caem de ~12.500 para ~10.400, porque a rede faz mais cálculo por imagem.
- **Não é reproduzível:** as linhas r1 e r2 têm a mesma semente e o mesmo código, mas resultados diferentes (0,9877 contra 0,9864; perda de validação 0,0490 contra 0,0588). A variação é de 13 imagens. Ao contrário do MLP, a convolução na GPU não é determinística, provavelmente por causa de algoritmos do cuDNN. Não verifiquei a causa.
- **Sobreajuste maior do que a acurácia sugere:** a diferença de acurácia entre treino e validação (0,9 a 1,0 ponto) é parecida com a do MLP, mas a perda de validação é 4,7 a 6,0 vezes a de treino (no MLP era 2,0). A CNN acerta mais, porém fica mais "confiante" nos erros que comete. Isso indica que regularização (*dropout*, *data augmentation*) ainda traria ganho.

### Evolução 1b — CNN em CPU (comparação de dispositivo)

| Linha | Acc. treino | Acc. validação | Erros | Perda validação | Tempo | Imagens/s |
|---|---|---|---|---|---|---|
| cnn-cpu-r1 | 0,9969 | 0,9852 | 148 | 0,0575 | 173,2 s | 1.732 |
| cnn-cpu-r2 | 0,9969 | 0,9852 | 148 | 0,0575 | 203,9 s | 1.471 |

- **Reprodutível na CPU:** as duas linhas têm resultado idêntico, o que indica que a variação da evolução 1 vem da GPU e não da semente.
- **GPU contra CPU:** a CNN leva 173 a 204 s em CPU contra ~29 s em GPU, ou seja, de 6 a 7 vezes mais. Os tempos da CPU diferem 18% entre as duas execuções; não investiguei o porquê.
- **Acurácia em CPU:** 0,9852 fica dentro da faixa observada na GPU (0,9864–0,9887) mais um pouco abaixo. Com apenas uma semente, não dá para afirmar que a CPU seja pior; a diferença é pequena e do tamanho da variação entre execuções.

### Evolução 2 — Cache em memória e threads (MLP)

| Linha | Cache | Threads | Acc. validação | Perda validação | Tempo | Imagens/s | Ganho vs. 22,8 s |
|---|---|---|---|---|---|---|---|
| otim-mlp-base-ref | não | 0 | 0,9765 | 0,0802 | 22,8 s | 13.130 | 1,0× |
| otim-mlp-cache | sim | 0 | 0,9765 | 0,0802 | 15,1 s | 19.857 | 1,5× |
| otim-mlp-threads8 | não | 8 | 0,9765 | 0,0802 | 11,6 s | 25.967 | 2,0× |
| otim-mlp-cache-threads8 | sim | 8 | 0,9765 | 0,0802 | 11,4 s | 26.330 | 2,0× |

- **O treino não mudou:** acurácias e perdas são idênticas em todas as linhas (0,9868, 0,9765, 0,0409, 0,0802). O cache usa a mesma conversão de pixels da leitura em disco (o teste da D8 garante isso). As otimizações mudam só o tempo, não o aprendizado.
- **O gargalo era o carregamento:** o tempo cai de 22,8 s para 11,4 s (2 vezes) sem alterar a rede.
- **Threads valem mais que cache:** só com threads o tempo é 11,6 s; só com cache, 15,1 s. Com as duas, 11,4 s. A diferença entre "threads" e "threads + cache" (0,2 s) está dentro da variação entre execuções, então o cache acrescenta pouco quando já há threads.
- **Por que parou em ~2 vezes:** 26.000 imagens por segundo é o limite para este modelo. O restante é cálculo e o custo de lançar operações pequenas na GPU, e o MLP mal ocupa a placa (a medição de `nvidia-smi` durante esta etapa deu uso médio de 7% a 8%).

### Evolução 2b — Cache e threads (CNN) e lote maior

| Linha | Cache | Threads | Lote | Acc. validação | Erros | Perda validação | Tempo | Imagens/s |
|---|---|---|---|---|---|---|---|---|
| cnn-r1, cnn-r2 (referência) | não | 0 | 32 | 0,9877 / 0,9864 | 123 / 136 | 0,0490 / 0,0588 | 28,7 / 28,9 s | ~10.400 |
| otim-cnn-cache | sim | 0 | 32 | 0,9887 | 113 | 0,0458 | 20,4 s | 14.672 |
| otim-cnn-cache-threads8 | sim | 8 | 32 | 0,9873 | 127 | 0,0453 | 14,8 s | 20.324 |
| otim-cnn-cache-lote128 | sim | 0 | 128 | 0,9880 | 120 | 0,0426 | 13,9 s | 21.510 |

- **Tempo:** cache sozinho dá 1,4 vez (28,8 s → 20,4 s); cache mais 8 threads, 1,9 vez (→ 14,8 s). O ganho é um pouco menor que no MLP porque na CNN uma parte maior do tempo é cálculo real.
- **Acurácia sem relação com as otimizações:** os valores 0,9887, 0,9873 e 0,9880 ficam todos na faixa da CNN em GPU sem otimização (0,9864–0,9877) e mais um pouco acima. Como a CNN em GPU não é determinística (evolução 1), essas diferenças de 1 a 2 décimos de ponto não devem ser atribuídas ao cache nem às threads. Com cinco execuções de CNN em GPU, a acurácia fica entre 0,9864 e 0,9887, uma faixa de 0,23 ponto, média de 0,9876.
- **Lote 128:** o tempo cai para 13,9 s (2,1 vezes em relação à referência) porque cada operação na GPU processa mais imagens. A acurácia (0,9880) ficou na mesma faixa. Esta linha mistura duas mudanças (cache e lote), então o ganho de tempo não é só do lote; o CSV não tem a CNN com lote 128 sem cache para separar.

### Evolução 3 — Avaliação em lote (fora do CSV)

A avaliação em lote não gera linha no `treinos.csv`, porque não treina. O resultado dela foi: CNN 98,80% (9.880 de 10.000) em `images-validation`, dígito 9 com 96,33% e os demais entre 98,2% e 99,7%. Esse número vem do caminho de inferência real (`DigitClassifier`), e não do avaliador interno do treino. Por isso é uma verificação independente do modelo salvo.

### Resumo da evolução

| Etapa | Mudança | Acc. validação | Erros | Tempo | Efeito |
|---|---|---|---|---|---|
| 0 | Baseline MLP, GPU | 0,9765 | 235 | ~24 s | Referência reprodutível |
| 1 | MLP → CNN, GPU | 0,9864–0,9877 | 123–136 | ~29 s | −45% de erros, +20% de tempo |
| 1b | CNN em CPU | 0,9852 | 148 | 173–204 s | GPU é 6–7× mais rápida |
| 2 | MLP + cache + 8 threads | 0,9765 | 235 | 11,4 s | 2× mais rápido, mesma acurácia |
| 2b | CNN + cache + 8 threads | 0,9873 | 127 | 14,8 s | 1,9× mais rápida, mesma faixa de acurácia |

Em conjunto, a CNN com cache e 8 threads atingiu a faixa de ~98,7–98,9% em cerca de 15 s, contra 97,65% em ~24 s do ponto de partida.

---

## Limites do que foi medido

- **Só MNIST:** os números valem para dígitos limpos, centralizados, brancos em fundo preto. Fotos reais não foram testadas, e o modelo não inverte cores nem localiza dígitos.
- **Validação também é o teste:** as 10.000 imagens servem para acompanhar o treino e para a nota final, então a acurácia é um pouco otimista.
- **Poucas repetições:** a maioria das configurações foi executada uma vez. A CNN em GPU varia em torno de 0,2 ponto, então diferenças menores que isso não são conclusivas.
- **Medições fora do CSV:** a primeira comparação CPU contra GPU do MLP (36,8 s contra 23,2 s com lote 32) e as execuções sem semente (97,5% a 97,8%) foram feitas antes de o CSV existir e constam só na conversa e nos relatórios parciais.
- **Uso da GPU:** as porcentagens de utilização (cerca de 7% a 8% no MLP, 28% a 38% na CNN) vieram de amostragens do `nvidia-smi` que incluem a inicialização da JVM, então subestimam o uso durante o treino. Elas não estão no CSV.

## Próximos passos sugeridos

1. Regularização (*dropout* e *data augmentation*) para reduzir o sobreajuste da CNN, visível na razão entre as perdas de validação e treino.
2. Separar um conjunto de teste do de validação.
3. Repetir cada configuração 3 a 5 vezes e registrar média e desvio, para tornar as comparações da CNN em GPU conclusivas.
4. Testar o modelo em imagens fora do MNIST, antes de qualquer uso real.
