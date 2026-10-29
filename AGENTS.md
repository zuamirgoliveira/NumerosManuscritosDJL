# AGENTS.md

Guia para agentes de código e novos contribuidores. Descreve o repositório **como o código realmente é** (verificado lendo o código-fonte, não o README). Divergências com o README estão na seção [README vs. realidade](#readme-vs-realidade).

## Visão geral

Projeto didático de Deep Learning em Java: reconhecimento de dígitos manuscritos (MNIST, 10 classes).

- Java 21, Maven, artefato `com.example:djl-mnist-example:1.0.0`
- Deep Java Library (DJL) 0.36.0 com engine **PyTorch** (`api`, `model-zoo`, `pytorch-engine`, via BOM)
- JavaFX 21.0.2 (`controls`, `graphics`, `base`) com classifier por SO
- `slf4j-simple` 2.0.13 para logging
- Idioma: **todo o código, comentários, Javadoc e mensagens ao usuário estão em português (pt-BR)**. Mantenha.
- Não há testes automatizados (`src/test` não existe).

Pipeline: `PrepareTrainingImages` → `TrainMnist` → `InferMnist` (CLI) ou `ui.InferenceApp` (JavaFX).

## Estrutura

```
pom.xml                              dependências, profiles por SO, exec e shade plugins
build/mnist-model/mnist-mlp-0005.params   modelo MLP treinado (versionado no git)
build/mnist-model/mnist-cnn-0005.params   modelo CNN treinado (versionado no git)
results/                             treinos.csv (1 linha por treino) e relatórios .md de cada experimento
src/main/java/com/example/mnist/
  DigitClassifier.java               constantes + arquitetura + carga do modelo + inferência
  ImageFolderDataset.java            Dataset DJL customizado ("uma pasta por classe")
  PrepareTrainingImages.java         baixa MNIST (IDX) e grava PNGs     [main]
  TrainMnist.java                    treina e salva o modelo            [main]
  InferMnist.java                    inferência por CLI                 [main]
  ui/Launcher.java                   Main-Class do fat JAR (delega ao app)
  ui/InferenceApp.java               GUI JavaFX                         [main via Launcher]
```

Diretórios gerados, ignorados pelo git: `images-training/` (60.000 PNGs) e `images-validation/` (10.000 PNGs), criados por `PrepareTrainingImages`.

## Comandos

Sempre executar **da raiz do projeto**: todos os caminhos (`images-training`, `images-validation`, `build/mnist-model`) são relativos ao diretório atual.

```bash
mvn clean compile
mvn exec:java -Dexec.mainClass="com.example.mnist.PrepareTrainingImages"   # uma vez; precisa de internet
mvn exec:java -Dexec.mainClass="com.example.mnist.TrainMnist" -Dexec.args="32 5"   # [lote] [épocas]
mvn exec:java -Dexec.mainClass="com.example.mnist.InferMnist" -Dexec.args="images-validation/7/00064.png"
mvn clean package && java -jar target/djl-mnist-example-1.0.0.jar          # GUI
```

**`exec:java` não recompila:** rode `mvn compile` depois de mudar o código. O `mvn` pode não estar no PATH (`/usr/share/maven/bin/mvn`).

### Opções do treino (propriedades `-D`)

| Opção | Efeito | Padrão |
|---|---|---|
| `-Dmodelo.arquitetura=mlp\|cnn` | Arquitetura (também vale para `InferMnist` e a GUI) | `mlp` |
| `-Dtreino.device=gpu\|cpu\|auto` | Força o device | `auto` |
| `-Dtreino.seed=N` | Semente (pesos e embaralhamento) | 42 |
| `-Dtreino.cache=true` | Decodifica os PNGs uma vez e os mantém em memória | false |
| `-Dtreino.threads=N` | Monta os lotes em N threads daemon | 0 (mesma thread) |
| `-Dtreino.rotulo=nome` | Nome da linha em `results/treinos.csv` | `sem-rotulo` |

**GPU:** o DJL 0.36 não detecta a GPU sem o CUDA Toolkit (`GPUs disponíveis: 0`) e baixa o PyTorch só-CPU. Neste ambiente (RTX 5050, Blackwell) é preciso `PYTORCH_FLAVOR=cu128` **e** `-Dtreino.device=gpu`.

## Componentes e regras de negócio

### `DigitClassifier` (fonte única de verdade do modelo)
- Constantes públicas: `IMAGE_HEIGHT=28`, `IMAGE_WIDTH=28`, `NUM_CLASSES=10`, `DEFAULT_MODEL_DIR=build/mnist-model`.
- `enum Architecture { MLP("mnist-mlp"), CNN("mnist-cnn") }`: cada uma com `modelName()` (nome do arquivo de pesos) e `newBlock()`. `fromProperty()` lê `-Dmodelo.arquitetura`.
  - MLP: `Mlp(784, 10, {128, 64})`.
  - CNN: `Conv 3x3(32) → ReLU → Conv 3x3(64) → ReLU → MaxPool 2x2 → Flatten → Linear 128 → ReLU → Linear 10`.
- **Treino e inferência usam `Architecture.newBlock()`**: a estrutura nunca deve ser definida em outro lugar. O `.params` guarda só pesos, então mudar a arquitetura exige retreinar.
- Construtores: `()` e `(Path)` usam `Architecture.fromProperty()`; `(Path, Architecture)` é o principal: `Model.newInstance` → `setBlock` → `model.load(dir, nome)` (DJL escolhe a época mais alta) → `Predictor`. Lança `IOException` se o modelo não existe e `MalformedModelException` se os pesos não casam.
- `classify(Path)`: `ImageFactory.fromFile` → `predictor.predict` → `Classifications`.
- `Translator`: entrada `GRAYSCALE` → `resize(28,28)` → `toTensor` (0–1, `(1,28,28)`); saída `softmax(0)` com classes `"0"`…`"9"`; `Batchifier.STACK`.
- `AutoCloseable`: **sempre usar try-with-resources** (memória nativa).
- Não há inversão de cores: o modelo espera dígito claro em fundo escuro, centralizado (estilo MNIST).

### `ImageFolderDataset` (extends `RandomAccessDataset`)
- Padrão **Builder** (`builder().setRoot(path).setSampling(batch, shuffle).build()`); `Builder extends BaseBuilder<Builder>` com `self()`.
- `prepare(Progress)` é idempotente (flag `prepared`). Considera apenas subpastas com nome **numérico** (`\d+`), ordenadas numericamente; **o nome da pasta é o rótulo**. Só indexa arquivos `.png` (case-insensitive), ordenados. Lança `IOException` com instrução para rodar `PrepareTrainingImages` se não achar subpastas.
- Com `setCacheInMemory(true)`, `prepare()` decodifica tudo em paralelo e guarda os bytes uint8 (~55 MB); `get()` só cria o tensor. Resultado idêntico ao caminho sem cache.
- Carregamento preguiçoso (sem cache): guarda `List<Path>` + `List<Integer>`; `get(manager, index)` lê um PNG, converte para `GRAYSCALE` + `toTensor`, rótulo como `float`. **Não redimensiona**: assume imagens já 28×28 (o `Translator` da inferência sim).
- `availableSize()` = nº de imagens indexadas.

### `PrepareTrainingImages`
- Baixa 4 `.gz` do espelho `MIRROR = https://ossci-datasets.s3.amazonaws.com/mnist/` (o site oficial tem problemas de certificado) com `java.net.http.HttpClient`; tudo em memória.
- Valida cabeçalho IDX (magic 2051 imagens / 2049 rótulos) e que nº de imagens == nº de rótulos.
- Grava `NNNNN.png` (`%05d`, índice global do arquivo) em `<saída>/<dígito>/`. Treino → `images-training`, teste `t10k` → `images-validation`.
- Não é idempotente: sobrescreve arquivos existentes. Imprime progresso a cada 10.000.

### `TrainMnist`
- Argumentos posicionais `[lote] [épocas]` (padrões 32 e 5) e as propriedades `-D` da tabela acima. Semente definida **antes** de criar modelo e datasets.
- Treino com shuffle, validação sem shuffle. Config: `Loss.softmaxCrossEntropyLoss()` + `Accuracy` + `TrainingListener.Defaults.logging()`. Otimizador/learning rate são os padrões do DJL.
- Sem `optExecutorService`, o DJL monta os lotes na thread do treino; com `treino.threads` usa um pool **daemon** (um pool não-daemon impede o programa de terminar).
- Imprime engine, GPUs, device, tempo por época e imagens/s; salva em `<modelName>-0005.params`; acrescenta uma linha a `results/treinos.csv` (data, rótulo, arquitetura, device, lote, épocas, semente, cache, threads, acurácias, perdas, tempos).
- Reprodutibilidade: MLP (GPU) e CNN (CPU) repetem o resultado com a mesma semente; **CNN na GPU varia ~0,1–0,2 ponto** (não determinismo do cuDNN).
- Nota: `model` não está em try-with-resources (vaza se houver exceção).

### `InferMnist`
CLI mínima. Sem argumento imprime o uso e retorna (sem erro). Com argumento imprime as probabilidades e `>>> Dígito reconhecido: X (confiança YY.YY%)`.

### `ui/InferenceApp` (JavaFX)
- Carrega `DigitClassifier` **uma vez** em `start()`. `IOException` → alerta fatal com instruções para preparar/treinar e `Platform.exit()`; outra `Exception` → alerta fatal genérico.
- Fluxo: "Selecionar imagem..." (`FileChooser`: png/jpg/jpeg/bmp) habilita "Classificar dígito". Classificação roda em `javafx.concurrent.Task` numa thread daemon `digit-classification`, com `ProgressIndicator`; sucesso → `showResults`, falha → `showError`.
- `showResults`: dígito vencedor em destaque + 10 barras (`ProgressBar`) ordenadas por probabilidade decrescente.
- Fecha o classificador em `setOnCloseRequest`. UI montada programaticamente (sem FXML/CSS externo; só `setStyle` inline). Janela 420×580.

### `ui/Launcher`
`Main-Class` do JAR. **Não** estende `Application`, apenas chama `InferenceApp.main(args)`; necessário porque um fat JAR com `Main-Class` que estende `Application` falha com "JavaFX runtime components are missing". Não alterar isso.

## Build (`pom.xml`)

- `maven.compiler.release=21`, UTF-8. Propriedade `app.mainClass=com.example.mnist.ui.Launcher`.
- Profiles `windows`/`mac`/`linux` (ativação por SO) definem `javafx.platform` (`win`/`mac`/`linux`; padrão `linux`) usada como `classifier` das dependências JavaFX → **o fat JAR só roda no SO em que foi gerado**.
- `maven-shade-plugin` 3.5.1 na fase `package`, com `ServicesResourceTransformer` (necessário para o DJL achar o engine PyTorch) e `ManifestResourceTransformer`. Gera `dependency-reduced-pom.xml`.
- `exec-maven-plugin` 3.3.0 para rodar as classes `main`.
- Binários nativos do PyTorch são baixados pelo DJL na primeira execução (precisa de internet).

## Padrões e convenções

- Classes utilitárias/entrypoints: `public final class` com construtor `private` vazio (`TrainMnist`, `InferMnist`, `PrepareTrainingImages`, `Launcher`).
- Cada etapa do pipeline é uma classe com `main` independente, executada via `exec:java` (não há CLI unificada nem framework de args).
- Constantes de configuração como `static final` no topo da classe (sem arquivo de configuração, sem env vars).
- Reuso: treino, CLI e GUI compartilham `DigitClassifier`; pré-processamento do treino (`ImageFolderDataset`) e da inferência (`Translator`) devem permanecer equivalentes (cinza + `toTensor`, 0–1).
- Recursos nativos (`Model`, `Predictor`, `Trainer`, `NDManager`) via try-with-resources/`AutoCloseable`.
- Erros: exceções checadas propagadas nos `main`; mensagens de erro em português e acionáveis (dizem qual comando rodar).
- Javadoc e comentários abundantes e didáticos em português, explicando o "porquê"; siga essa densidade ao editar.
- Nenhuma dependência de `ai.djl:basicdataset` (de propósito); o dataset é próprio.

## Resultados medidos (MNIST, lote 32, 5 épocas, semente 42, GPU)

MLP 97,65% (~23 s) → CNN ~98,5–98,8% (~29 s). Com `cache` + 8 threads: MLP 11,4 s, CNN 14,8 s (~2×), mesma acurácia. Detalhes em `results/*.md`. Medido só em imagens do MNIST, não em imagens reais.

## Armadilhas

- Rodar fora da raiz quebra os caminhos relativos.
- Foto de dígito escuro em fundo claro será classificada errado (sem inversão automática).
- Mudou `Architecture.newBlock()`/tamanho de imagem → retreinar e regerar o `.params` correspondente.
- Os `.params` estão versionados e cada treino os sobrescreve; retreinar com outro nº de épocas gera outro arquivo (`load` pega o de maior época, então um arquivo antigo de época maior sobrepõe o novo).
- `target/` está parcialmente versionado por engano; não inclua alterações dele nos commits.

## README vs. realidade

O `README.md` descreve corretamente classes, arquitetura, fluxo e pom. Divergências encontradas:

1. **Pastas de imagens**: o README diz "se as pastas já existem (como neste projeto), pule este passo" e as lista na árvore. Elas **não existem no repositório**; é necessário rodar `PrepareTrainingImages` antes de treinar ou usar os exemplos de `images-validation/`.
2. **`dependency-reduced-pom.xml`**: o README diz que está no `.gitignore`. **Não está**, e o arquivo está versionado. O mesmo vale para `target/` (parcialmente versionado: `maven-archiver`, `maven-status`) e `.idea/`. O `.gitignore` é o template padrão de Java (ignora `*.jar`, `*.class`, etc.).
3. **Árvore do projeto**: omite `build/` como versionado, `LICENSE`, `.idea/` e `dependency-reduced-pom.xml`; o nome raiz `djl-mnist-example/` difere do diretório do repositório (`NumerosManuscritosDJL`).
4. **Acurácia "97–98%"** e "5 épocas/lote 32" não foram verificadas por execução; os valores de épocas/lote batem com o código, a acurácia é só estimativa do README.
5. A seção "Ideias para evoluir" menciona `LeNet` do model-zoo e engine MXNet; nada disso está no código atual.
