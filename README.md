# Reconhecimento de Dígitos Manuscritos com DJL + JavaFX

Projeto didático de ponta a ponta de **Deep Learning em Java**: prepara o
dataset MNIST como imagens PNG, **treina** uma rede neural (MLP ou CNN),
**avalia** o resultado e permite **classificar dígitos** pela **linha de
comando** ou por uma **interface gráfica desktop (JavaFX)**.

Construído com [Deep Java Library (DJL) 0.36.0](https://djl.ai/) (engine
PyTorch), Java 21 e Maven.

> O projeto **não** usa o módulo `ai.djl:basicdataset`. Os dados viram
> arquivos `.png` reais em disco e são lidos por um `Dataset` customizado
> (`ImageFolderDataset`) — o mesmo padrão "uma pasta por classe" que você
> usaria com suas próprias imagens.

---

## Sumário

1. [Visão geral](#visão-geral)
2. [Pré-requisitos](#pré-requisitos)
3. [Estrutura do projeto](#estrutura-do-projeto)
4. [Como executar](#como-executar)
5. [Opções do treino](#opções-do-treino)
6. [Como funciona](#como-funciona)
7. [Explicação das classes](#explicação-das-classes)
8. [Decisões de projeto e por que foram melhores](#decisões-de-projeto-e-por-que-foram-melhores)
9. [Configuração (`pom.xml`)](#configuração-pomxml)
10. [Dicas e limitações](#dicas-e-limitações)
11. [Solução de problemas](#solução-de-problemas)
12. [Ideias para evoluir](#ideias-para-evoluir)

---

## Visão geral

```
 PrepareTrainingImages     TrainMnist                 InferMnist (1 imagem)
 baixa o MNIST e grava  →  treina (MLP ou CNN),   →   EvaluateMnist (pasta inteira)
 PNGs em pastas            salva o modelo e           InferenceApp (JavaFX)
                           registra o resultado
```

| Item | Valor |
|---|---|
| Tarefa | Classificação de imagens em 10 classes (dígitos 0–9) |
| Dataset | MNIST — 60.000 imagens de treino + 10.000 de validação (28×28, tons de cinza) |
| Modelos | **MLP** `784 → 128 → 64 → 10` (padrão) e **CNN** `Conv 3×3(32) → Conv 3×3(64) → MaxPool → Linear 128 → 10` |
| Treino | 5 épocas, lotes de 32, perda *softmax cross-entropy*, semente fixa (42) |
| Engine | PyTorch (via DJL), CPU ou GPU |

Resultados medidos neste repositório (MNIST, lote 32, 5 épocas, semente 42):

| Modelo | Acurácia de validação | Tempo de treino (GPU) | Com `cache` + 8 threads |
|---|---|---|---|
| MLP | 97,65% | ~23 s | 11,4 s |
| CNN | ~98,5–98,8% | ~29 s | 14,8 s |

Os relatórios completos estão em [`results/`](results/).

---

## Pré-requisitos

- **JDK 21** (`java -version` deve mostrar 21.x)
- **Maven 3.8+** (`mvn -version`)
- **Internet** na primeira execução, para baixar o MNIST e os binários
  nativos do PyTorch e do JavaFX.
- **GPU (opcional):** placa NVIDIA com driver instalado. Não é preciso instalar
  o CUDA Toolkit (veja [Treinar na GPU](#treinar-na-gpu)).

---

## Estrutura do projeto

```
├── pom.xml                       # dependências, profiles (SO e gpu), testes e fat JAR
├── README.md
├── AGENTS.md                     # guia técnico para agentes de código e contribuidores
├── build/mnist-model/            # modelos treinados (versionados)
│   ├── mnist-mlp-0005.params
│   └── mnist-cnn-0005.params
├── results/                      # histórico dos experimentos
│   ├── treinos.csv               #   uma linha por treino (config, acurácia, tempo)
│   └── relatorio-*.md            #   relatórios: baseline, MLP x CNN, desempenho
├── images-training/              # 60.000 PNGs (gerado; fora do git)
├── images-validation/            # 10.000 PNGs (gerado; fora do git)
└── src/
    ├── main/java/com/example/mnist/
    │   ├── PrepareTrainingImages.java   # baixa o MNIST cru e grava os PNGs
    │   ├── ImageFolderDataset.java      # Dataset customizado (com cache opcional em memória)
    │   ├── DigitClassifier.java         # arquiteturas (MLP/CNN) + carregamento + inferência
    │   ├── TrainMnist.java              # treina, mede e salva o modelo
    │   ├── InferMnist.java              # inferência por linha de comando (1 imagem)
    │   ├── EvaluateMnist.java           # avaliação em lote: acurácia e matriz de confusão
    │   └── ui/
    │       ├── Launcher.java            # Main-Class real do JAR
    │       └── InferenceApp.java        # interface gráfica JavaFX
    └── test/java/com/example/mnist/     # testes JUnit 5
```

---

## Como executar

Execute os comandos **sempre na raiz do projeto** (a pasta do `pom.xml`): os
caminhos `images-training/`, `images-validation/` e `build/mnist-model/` são
relativos ao diretório atual.

> `mvn exec:java` **não recompila**. Rode `mvn compile` depois de alterar o
> código.

### 1. Compilar e testar

```bash
mvn clean compile
mvn test            # 13 testes, não exigem as imagens nem GPU
```

### 2. Gerar as imagens (uma única vez)

```bash
mvn exec:java -Dexec.mainClass="com.example.mnist.PrepareTrainingImages"
```

Baixa o MNIST original e grava `images-training/<dígito>/*.png` (60.000) e
`images-validation/<dígito>/*.png` (10.000). Leva alguns minutos. Essas pastas
**não ficam no repositório** (estão no `.gitignore`), então este passo é
necessário para treinar, avaliar e para usar imagens de exemplo.

### 3. Treinar

```bash
# MLP (padrão), 32 imagens por lote, 5 épocas
mvn exec:java -Dexec.mainClass="com.example.mnist.TrainMnist" -Dexec.args="32 5"

# CNN
mvn exec:java -Dexec.mainClass="com.example.mnist.TrainMnist" \
    -Dexec.args="32 5" -Dmodelo.arquitetura=cnn
```

O console mostra o dispositivo usado (CPU/GPU), a perda e a acurácia de cada
época e um resumo de desempenho (tempo por época e imagens/s). Cada execução
acrescenta uma linha a `results/treinos.csv`. O modelo é salvo em
`build/mnist-model/` com o nome da arquitetura (`mnist-mlp-…` ou `mnist-cnn-…`).

> Os modelos treinados já estão no repositório: se quiser só usar, pule este
> passo.

#### Treinar na GPU

```bash
mvn exec:java -Pgpu -Dexec.mainClass="com.example.mnist.TrainMnist" \
    -Dexec.args="32 5" -Dmodelo.arquitetura=cnn \
    -Dtreino.cache=true -Dtreino.threads=8
```

O perfil `-Pgpu` configura o DJL para baixar o PyTorch com CUDA 12.8 e usar a
GPU 0 (veja a [decisão 6](#6-perfil-maven--pgpu-em-vez-de-passo-manual)).

### 4. Avaliar o modelo em uma pasta inteira

```bash
mvn exec:java -Dexec.mainClass="com.example.mnist.EvaluateMnist" \
    -Dmodelo.arquitetura=cnn
# ou outra pasta (mesmo formato: subpastas 0..9):
mvn exec:java -Dexec.mainClass="com.example.mnist.EvaluateMnist" -Dexec.args="minha/pasta"
```

Imprime a acurácia geral, a acurácia **por dígito** e a **matriz de confusão**
(linha = dígito real, coluna = dígito previsto).

### 5. Classificar um dígito

**a) Linha de comando (uma imagem)**

```bash
mvn exec:java -Dexec.mainClass="com.example.mnist.InferMnist" \
    -Dexec.args="images-validation/7/00064.png" -Dmodelo.arquitetura=cnn
```

Saída: as probabilidades de cada dígito e o dígito reconhecido.

**b) Interface gráfica (JavaFX)**

```bash
mvn clean package
java -jar target/djl-mnist-example-1.0.0.jar
# para usar a CNN:
java -Dmodelo.arquitetura=cnn -jar target/djl-mnist-example-1.0.0.jar
```

Na janela:

1. Clique em **Selecionar imagem...** e escolha um `.png`/`.jpg`/`.bmp`
   (por exemplo, qualquer arquivo de `images-validation/<dígito>/`).
2. A imagem aparece na tela.
3. Clique em **Classificar dígito**: o dígito reconhecido aparece em
   destaque, com uma barra de probabilidade para cada classe (0–9).

A interface classifica **uma imagem por vez**; para medir uma pasta inteira use
o `EvaluateMnist`.

---

## Opções do treino

Passadas com `-D` no Maven (por exemplo `-Dtreino.seed=7`). Argumentos
posicionais do `TrainMnist` (via `-Dexec.args`): `[tamanho do lote] [épocas]`
(padrão `32 5`).

| Opção | Efeito | Padrão |
|---|---|---|
| `-Dmodelo.arquitetura=mlp\|cnn` | Arquitetura (vale também para `InferMnist`, `EvaluateMnist` e a GUI) | `mlp` |
| `-Dtreino.device=gpu\|cpu\|auto` | Força o dispositivo | `auto` |
| `-Dtreino.seed=N` | Semente (pesos e embaralhamento) | `42` |
| `-Dtreino.cache=true` | Decodifica os PNGs uma vez e os mantém em memória | `false` |
| `-Dtreino.threads=N` | Monta os lotes em N threads, em paralelo ao treino | `0` (mesma thread) |
| `-Dtreino.rotulo=nome` | Nome da execução na linha do `results/treinos.csv` | `sem-rotulo` |
| `-Pgpu` (perfil Maven) | CUDA 12.8 + GPU 0 | desligado |

---

## Como funciona

1. **Preparo dos dados** — `PrepareTrainingImages` lê os arquivos binários
   do MNIST (formato IDX) e grava cada imagem como PNG em
   `images-*/<dígito>/`. O **nome da pasta é o rótulo**.
2. **Leitura no treino** — `ImageFolderDataset` indexa os caminhos dos
   arquivos e converte cada imagem em tensor `(1, 28, 28)` com valores entre 0
   e 1. Por padrão decodifica cada PNG sob demanda; com `-Dtreino.cache=true`
   decodifica tudo uma única vez e guarda na memória.
3. **Treino** — `TrainMnist` fixa a semente, ajusta os pesos em lotes, mede a
   acurácia no conjunto de validação, imprime o desempenho e registra a
   execução em `results/treinos.csv`.
4. **Inferência** — `DigitClassifier` recria **a mesma arquitetura** usada no
   treino (`Architecture.newBlock()`), carrega os pesos salvos e, para cada
   imagem: converte para cinza → redimensiona para 28×28 → normaliza → roda a
   rede → aplica `softmax` (probabilidades).

O pré-processamento do treino e o da inferência são equivalentes; isso é
essencial para o modelo funcionar bem fora do treino.

---

## Explicação das classes

### `PrepareTrainingImages`
Baixa os 4 arquivos `.gz` do MNIST de um espelho público, valida o
cabeçalho IDX (números mágicos 2051/2049) e grava as imagens como PNG. O
site oficial (`yann.lecun.com`) costuma ter problemas de certificado, por
isso usa-se o espelho `https://ossci-datasets.s3.amazonaws.com/mnist/`
(constante `MIRROR`, fácil de trocar se ficar indisponível).

### `ImageFolderDataset`
Subclasse de `RandomAccessDataset` do DJL, com o padrão *builder*:

```java
ImageFolderDataset ds = ImageFolderDataset.builder()
        .setRoot(Paths.get("images-training"))
        .setSampling(32, true)       // tamanho do lote + embaralhar
        .setCacheInMemory(true)      // opcional: decodifica tudo uma vez
        .build();
ds.prepare(new ProgressBar());
```

| Método | Função |
|---|---|
| `prepare()` | Lista os `.png` de cada subpasta numérica e guarda (caminho, rótulo). Ignora pastas com nome não numérico e ordena os arquivos para manter a ordem determinística. Com cache, também decodifica todas as imagens (em paralelo). |
| `get(index)` | Devolve o par (tensor, rótulo): lê o PNG do disco ou usa os bytes em memória (resultado idêntico). |
| `availableSize()` | Total de imagens indexadas. |

Para usar com **outro dataset seu**, basta a mesma estrutura de pastas
numeradas (`0/`, `1/`, …).

### `DigitClassifier`
"Fonte única de verdade" do modelo — usada por treino, CLI, avaliação e interface:

| Membro | Função |
|---|---|
| `enum Architecture` | `MLP` e `CNN`. Cada uma tem `modelName()` (nome do arquivo de pesos) e `newBlock()` (a rede). `fromProperty()` lê `-Dmodelo.arquitetura`. Compartilhada com o treino para nunca divergirem. |
| Construtores | Criam o modelo, aplicam `newBlock()` e carregam os pesos de `build/mnist-model`. `(Path, Architecture)` é o principal. |
| `classify(Path)` | Lê a imagem e devolve as probabilidades (`Classifications`). |
| `Translator` interno | Pré-processa a imagem (cinza, 28×28, normalização) e pós-processa a saída (`softmax`). |
| `close()` | Libera memória nativa (`AutoCloseable` → use *try-with-resources*). |

### `TrainMnist`
Monta os datasets, cria o modelo da arquitetura escolhida, configura o treino
(perda `softmaxCrossEntropyLoss` + métrica `Accuracy`), executa
`EasyTrain.fit(...)`, salva o resultado e registra a execução em
`results/treinos.csv`. Lote e épocas vêm de argumentos; o resto, das opções
`-D` da [tabela](#opções-do-treino). O `Model` é aberto em
*try-with-resources*, então é fechado mesmo se o treino falhar.

### `EvaluateMnist`
Avaliação em lote: percorre uma pasta (padrão `images-validation`), classifica
cada imagem e imprime acurácia geral, acurácia por dígito e matriz de confusão.

### `InferMnist`
CLI mínima: lê o caminho de **uma** imagem, chama `DigitClassifier` e imprime o
resultado.

### `ui/InferenceApp`
Aplicação JavaFX:

| Método | Função |
|---|---|
| `start(Stage)` | Carrega o modelo uma vez (mostra alerta claro se ainda não foi treinado) e monta a janela. |
| `onSelectImage()` | Abre o seletor de arquivos nativo e exibe a imagem escolhida. |
| `onClassify()` | Roda a inferência numa `Task` (outra thread), para a janela **não travar**. |
| `showResults()` | Mostra o dígito em destaque + barras de probabilidade 0–9. |

### `ui/Launcher`
É o `Main-Class` do JAR. Existe por causa de uma particularidade do JavaFX:
se o `Main-Class` estende `javafx.application.Application`, o
`java -jar` falha com *"JavaFX runtime components are missing"* em um fat
JAR (classpath simples). O `Launcher` **não** estende `Application` e só
delega para `InferenceApp.main(args)`, contornando o problema.

---

## Decisões de projeto e por que foram melhores

Cada decisão abaixo foi tomada com base em medições feitas neste repositório
(MNIST completo, GPU RTX 5050 Laptop, 5 épocas, semente 42, salvo indicação). Os
números brutos estão em [`results/treinos.csv`](results/treinos.csv) e os
relatórios em [`results/`](results/).

### 1. Semente fixa e registro de cada treino em CSV

- **O problema:** sem semente, a mesma configuração dava 97,5%, 97,8% e 97,7%
  em execuções diferentes. Com ~0,3 ponto de ruído, não dá para saber se uma
  mudança melhorou alguma coisa.
- **A decisão:** semente padrão 42 (inicialização dos pesos e embaralhamento),
  definida antes de criar modelo e datasets; cada treino acrescenta uma linha a
  `results/treinos.csv` (configuração, acurácias, perdas, tempos).
- **Por que foi melhor:** duas execuções do MLP deram resultados **idênticos**
  (97,65%, mesmas perdas), então qualquer diferença posterior é efeito da
  mudança, não do acaso. O CSV dá uma linha de base comparável.
- **Limite conhecido:** a CNN na GPU **não** é totalmente reproduzível (98,64% a
  98,87% com a mesma semente), provavelmente por algoritmos de convolução não
  determinísticos do cuDNN. Na CPU ela é idêntica (98,52% nas duas execuções).
  Para comparações finas, use a CPU ou tire a média de várias execuções.

### 2. CNN no lugar do MLP (mantendo as duas)

- **O problema:** o MLP achata a imagem em 784 números soltos e não sabe quais
  pixels são vizinhos; perde a noção de traços e curvas.
- **A decisão:** adicionar uma CNN (duas convoluções 3×3, *max pooling* e uma
  camada densa) e manter o MLP como opção.
- **Por que foi melhor:** a acurácia subiu de **97,65% para ~98,5–98,8%**; os
  erros em 10.000 imagens caíram de cerca de 235 para 125–150 (~40% a menos),
  mais que o ruído medido, a um custo de ~20% mais tempo na GPU (29 s contra
  24 s). Dígitos são formas, justamente o que filtros convolucionais detectam.
- **Ressalvas:** a CNN passa a depender da GPU (na CPU leva 173–204 s, de 6 a 7
  vezes mais que na GPU), e a melhoria foi medida **só no MNIST**, não em fotos
  reais. O sobreajuste continua (99,6% de acurácia no treino contra 98,7% na
  validação).

### 3. Uma arquitetura por nome (`Architecture`), cada uma com seu arquivo de pesos

- **O problema:** com uma única arquitetura fixa e um único nome de modelo,
  treinar a CNN sobrescreveria o baseline, impossibilitando comparar.
- **A decisão:** um `enum Architecture` em `DigitClassifier`, em que cada valor
  define a rede **e** o nome do arquivo (`mnist-mlp-…`, `mnist-cnn-…`); a
  escolha vem de `-Dmodelo.arquitetura`.
- **Por que foi melhor:** os dois modelos coexistem e podem ser avaliados lado a
  lado. A arquitetura continua definida num só lugar, usada por treino e
  inferência, o que evita o erro clássico de treinar com uma estrutura e
  carregar os pesos em outra. O padrão continua `mlp`, então a GUI e a CLI não
  mudaram de comportamento.

### 4. Cache em memória e carregamento em threads

- **O problema:** o MLP é tão pequeno que o tempo ia quase todo na leitura e
  decodificação de PNGs. Sem `optExecutorService`, o DJL monta os lotes na
  própria thread do treino, em sequência, e a GPU ficava ociosa (~7–8% de uso
  médio com o MLP).
- **A decisão:** `-Dtreino.cache=true` (decodifica as 70.000 imagens uma vez, ~55
  MB) e `-Dtreino.threads=N` (monta os lotes em N threads daemon, com
  pré-busca).
- **Por que foi melhor:**

  | Modelo | Sem otimização | Cache + 8 threads | Ganho |
  |---|---|---|---|
  | MLP | ~23 s | 11,4 s | ~2,0× |
  | CNN | ~29 s | 14,8 s | ~1,9× |

  A acurácia do MLP ficou **idêntica** (97,65%): o cache usa exatamente a mesma
  conversão dos pixels, então só o tempo muda. As threads rendem mais que o
  cache sozinho (MLP: 11,6 s só com threads, 15,1 s só com cache).
- **Por que desligadas por padrão:** para preservar o comportamento original
  (didático, baixo consumo de memória). Recomendamos ligá-las ao treinar.
- **Um detalhe que custou caro:** o pool de threads precisa ser *daemon*. Com um
  pool comum, o treino terminava e gravava o resultado, mas o processo nunca
  encerrava.

### 5. Escolher o dispositivo explicitamente

- **O problema:** o DJL 0.36 não detectou a GPU (`GPUs disponíveis: 0`) porque
  não há CUDA Toolkit instalado, e o treino rodava silenciosamente em CPU.
- **A decisão:** `-Dtreino.device=gpu|cpu|auto` e um log no início do treino com
  o engine, o número de GPUs e o dispositivo usado.
- **Por que foi melhor:** deixa claro onde o treino está rodando e permite
  forçar a GPU mesmo sem detecção automática, além de comparar CPU e GPU nas
  mesmas condições. Com a GPU forçada, o MLP (lote 32) passou de 36,8 s para
  23,2 s (1,6×); a CNN, de ~173–204 s para ~29 s.

### 6. Perfil Maven `-Pgpu` em vez de passo manual

- **O problema:** usar a GPU exigia exportar `PYTORCH_FLAVOR=cu128` **e** passar
  o dispositivo, um conhecimento fácil de esquecer e que não ficava registrado
  em lugar nenhum. A RTX 5050 (arquitetura Blackwell) exige o build CUDA 12.8 ou
  mais novo.
- **A decisão:** um profile `gpu` no `pom.xml` que configura o
  `exec-maven-plugin` com essas duas propriedades; basta `mvn exec:java -Pgpu ...`.
- **Por que foi melhor:** o passo a passo fica versionado e reproduzível, sem
  variáveis de ambiente. Para outra GPU, troque o valor `cu128` no profile.
- **Atenção:** uma variável de ambiente `PYTORCH_FLAVOR` (mesmo vazia) tem
  prioridade sobre a propriedade do perfil.

### 7. Avaliação em lote (`EvaluateMnist`) com matriz de confusão

- **O problema:** a CLI e a GUI classificam uma imagem por vez, e a acurácia
  geral esconde onde o modelo erra.
- **A decisão:** uma classe que avalia uma pasta inteira e imprime a acurácia
  geral, por dígito e a matriz de confusão.
- **Por que foi melhor:** mede o modelo pelo mesmo caminho de inferência usado em
  produção (não só pelo avaliador interno do treino) e mostra problemas
  concretos: na CNN, o dígito **9** é o mais fraco (96,33%), confundido
  principalmente com o 5 e o 4. A avaliação da CNN deu 98,80% (9.880 de 10.000).

### 8. Testes automatizados

- **O problema:** não havia testes; qualquer refatoração (como as acima) era
  verificada só rodando o treino inteiro.
- **A decisão:** 13 testes JUnit 5 (`mvn test`), que **não** precisam das imagens
  do MNIST nem de GPU: criam PNGs sintéticos e usam os modelos versionados.
- **Por que foi melhor:** o teste "cache igual ao disco" garante que a
  otimização do item 4 não altera os dados; os de arquitetura garantem a saída
  `1×10` das duas redes; os de classificação garantem que as probabilidades
  somam 1 e que uma pasta sem modelo falha com `IOException`.

### 9. Fechar o `Model` mesmo em caso de erro

- **O problema:** o modelo guarda memória nativa e só era fechado no fim do
  fluxo normal; uma exceção durante o treino o deixava aberto.
- **A decisão:** `try (Model model = ...)` no `TrainMnist`.
- **Por que foi melhor:** consistente com o resto do projeto (`Trainer`,
  `DigitClassifier`), sem custo e sem vazamento de recursos nativos.

### 10. Higiene do repositório

- **A decisão:** `images-training/` e `images-validation/` (70.000 PNGs) e
  `target/` (saída do Maven) fora do git; os modelos treinados e os relatórios
  versionados.
- **Por que foi melhor:** o repositório fica pequeno e o `git status` limpo; os
  dados são regenerados por um comando, e os resultados dos experimentos ficam
  registrados em `results/`.

---

## Configuração (`pom.xml`)

| Elemento | Por quê |
|---|---|
| BOM do DJL (`ai.djl:bom`) | Fixa em um só lugar versões compatíveis de todos os módulos DJL. |
| `api`, `model-zoo`, `pytorch-engine` | API do DJL, o bloco `Mlp` pronto e o engine que executa as operações (binários nativos baixados na 1ª execução). |
| `javafx-*` com `classifier ${javafx.platform}` | O JavaFX publica bibliotecas nativas por SO. |
| *Profiles* `windows` / `mac` / `linux` | Detectam o SO e ajustam `javafx.platform` automaticamente. |
| *Profile* `gpu` (só com `-Pgpu`) | Configura o `exec-maven-plugin` com `PYTORCH_FLAVOR=cu128` e `treino.device=gpu`. |
| `junit-jupiter` e `maven-surefire-plugin` | Testes automatizados (`mvn test`). |
| `exec-maven-plugin` | Permite `mvn exec:java -Dexec.mainClass=...`. |
| `maven-shade-plugin` | Gera o *fat JAR* (`java -jar`) na fase `package`. |
| `ServicesResourceTransformer` | Mescla os `META-INF/services`, sem o qual o DJL não acha o engine PyTorch dentro do JAR. |
| `ManifestResourceTransformer` | Define `ui.Launcher` como `Main-Class`. |

O arquivo `dependency-reduced-pom.xml` é gerado automaticamente pelo
shade-plugin e pode ser ignorado (hoje ele está versionado; não está no
`.gitignore`).

---

## Dicas e limitações

- **Fundo e traço:** o MNIST tem dígitos **brancos em fundo preto**. Uma foto
  de um número escuro em papel branco será lida "invertida" e provavelmente
  errada. Inverta as cores da imagem antes (ou treine com dados no mesmo
  estilo). Imagens de `images-validation/` funcionam direto.
- **Imagem centralizada, um dígito:** o modelo classifica a **imagem inteira como
  um único dígito**. Não localiza nem separa vários números numa mesma imagem.
- **Medido só no MNIST:** as acurácias acima valem para o MNIST (limpo e
  centralizado). O desempenho em fotos reais não foi medido.
- **Validação = teste:** o conjunto de 10.000 imagens é usado para acompanhar o
  treino e também para a nota final, então a acurácia é um pouco otimista.
- **JAR por sistema operacional:** as bibliotecas nativas (JavaFX e PyTorch)
  variam por SO; o JAR só roda no sistema em que foi gerado. Gere um JAR em
  cada SO de destino.
- **Diretório de execução:** rode sempre da raiz do projeto (caminhos
  relativos).
- **Mudou a arquitetura?** Altere apenas `DigitClassifier.Architecture.newBlock()`
  e **retreine**; pesos antigos não são compatíveis com outra estrutura.
- **Épocas e arquivo de pesos:** o sufixo do `.params` é o número de épocas
  (`-0005`). O DJL carrega o arquivo de **maior** época; um treino com menos
  épocas não substitui, na inferência, um arquivo de época maior que já exista.

---

## Solução de problemas

| Sintoma | Causa provável / solução |
|---|---|
| Alerta "Não foi possível carregar o modelo treinado" | Faltam os pesos da arquitetura escolhida: rode `TrainMnist` (com o mesmo `-Dmodelo.arquitetura`). |
| `Nenhuma subpasta de classe encontrada em ...` | Rode `PrepareTrainingImages` na raiz do projeto. |
| `Error: JavaFX runtime components are missing` | Execute pelo JAR gerado (`Launcher`), não por `InferenceApp` diretamente. |
| Erro de download do MNIST | Espelho fora do ar: troque a constante `MIRROR` em `PrepareTrainingImages`. |
| JAR não abre em outro computador | Sistema operacional diferente: gere o JAR naquele SO. |
| Resultado errado em foto própria | Veja "Fundo e traço" acima. |
| Log mostra `GPUs disponíveis: 0` e `Treinando em: [cpu()]` | Use `-Pgpu` (ou `-Dtreino.device=gpu` com `PYTORCH_FLAVOR` adequado). |
| `Could not run 'aten::empty_strided' ... 'CUDA' backend` | Pediu GPU mas o PyTorch carregado é o só-CPU: use `-Pgpu` e confira se não há `PYTORCH_FLAVOR` vazio no ambiente. |
| Alterei o código e nada mudou no `exec:java` | O `exec:java` não recompila: rode `mvn compile`. |
| O treino termina mas o processo não encerra | Pool de threads não-daemon (já corrigido no projeto); se criar outro, use threads daemon. |

---

## Ideias para evoluir

- **Regularização:** *dropout* e *data augmentation* (rotações e deslocamentos)
  para reduzir o sobreajuste (99,6% no treino contra 98,7% na validação).
- **Separar validação e teste**, para uma nota final sem viés.
- **Testar em imagens reais** (fora do MNIST) e então adicionar **inversão
  automática de cores** e **centralização** do dígito antes de classificar.
- Adicionar um **canvas** na interface para desenhar o dígito com o mouse, e uma
  versão em lote na GUI.
- **Ler vários dígitos** numa imagem, com uma etapa de segmentação antes do
  modelo.
- **Parada antecipada** e salvar o melhor modelo, em vez de só o do final.
- Usar o engine **MXNet** (mais leve em CPU): troque `pytorch-engine` por
  `ai.djl.mxnet:mxnet-engine` no `pom.xml`; o código Java não muda.
