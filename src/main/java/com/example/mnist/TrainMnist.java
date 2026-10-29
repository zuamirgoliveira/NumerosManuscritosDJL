package com.example.mnist;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;

/**
 * Treina uma rede neural (MLP ou CNN, ver {@link DigitClassifier.Architecture}) para reconhecer dígitos manuscritos.
 *
 * <p>Este exemplo NÃO usa o módulo {@code ai.djl:basicdataset} (e a classe
 * pronta Mnist). Em vez disso, lê as imagens diretamente das pastas
 * {@code images-training} e {@code images-validation}, geradas previamente por
 * {@link PrepareTrainingImages}, através do dataset customizado
 * {@link ImageFolderDataset}.
 *
 * <p>Fluxo: dataset → modelo → configuração do treino → treino → salvar.
 * A arquitetura da rede e as constantes (nome do modelo, pasta de saída)
 * vêm de {@link DigitClassifier}, o que garante que a inferência carregue o
 * modelo exatamente com a mesma estrutura usada aqui.
 */
public final class TrainMnist {

    /** Tamanho de lote padrão: quantas imagens são processadas juntas a cada atualização dos pesos. */
    private static final int DEFAULT_BATCH_SIZE = 32;

    /** Número de épocas padrão: quantas vezes o conjunto de treino completo é percorrido. */
    private static final int DEFAULT_EPOCHS = 5;

    /** Semente padrão: fixa a inicialização dos pesos e o embaralhamento, para o treino ser reproduzível. */
    private static final int DEFAULT_SEED = 42;

    /** Arquivo onde cada execução acrescenta uma linha com configuração e resultados. */
    private static final Path RESULTS_FILE = Paths.get("results/treinos.csv");

    private TrainMnist() {}

    public static void main(String[] args) throws IOException, TranslateException {
        // Uso: TrainMnist [tamanhoDoLote] [épocas]. Sem argumentos, usa os padrões.
        int batchSize = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_BATCH_SIZE;
        int epochs = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_EPOCHS;

        // Opções por propriedade do sistema (-D...): semente e rótulo do experimento,
        // este último só identifica a linha no CSV de resultados (ex.: "baseline-mlp").
        int seed = Integer.getInteger("treino.seed", DEFAULT_SEED);
        String label = System.getProperty("treino.rotulo", "sem-rotulo");

        // Otimizações de carregamento de dados (desligadas por padrão):
        // -Dtreino.cache=true  decodifica os PNGs uma vez e os mantém na memória;
        // -Dtreino.threads=N   monta os lotes em N threads, em paralelo ao treino.
        boolean cache = Boolean.getBoolean("treino.cache");
        int threads = Integer.getInteger("treino.threads", 0);

        // A semente precisa ser definida ANTES de criar modelo e datasets.
        Engine.getInstance().setRandomSeed(seed);

        // Garante que a pasta de saída do modelo exista.
        Path modelDir = DigitClassifier.DEFAULT_MODEL_DIR;
        Files.createDirectories(modelDir);

        // 1) DATASETS
        // O ImageFolderDataset aponta para as pastas com os .png gerados por
        // PrepareTrainingImages. prepare() apenas lista os arquivos; cada
        // imagem só é lida do disco durante o treino, sob demanda.

        // Treino: lotes embaralhados (shuffle = true) a cada época, para que a
        // rede não "decore" a ordem das imagens.
        ImageFolderDataset trainingSet = ImageFolderDataset.builder()
                .setRoot(Paths.get("images-training"))
                .setSampling(batchSize, true)
                .setCacheInMemory(cache)
                .optPrefetchNumber(Math.max(2, threads))
                .build();
        trainingSet.prepare(new ProgressBar());

        // Validação: imagens que a rede NÃO vê no treino, usadas só para medir
        // a acurácia real. Não precisa embaralhar (shuffle = false).
        ImageFolderDataset validationSet = ImageFolderDataset.builder()
                .setRoot(Paths.get("images-validation"))
                .setSampling(batchSize, false)
                .setCacheInMemory(cache)
                .optPrefetchNumber(Math.max(2, threads))
                .build();
        validationSet.prepare(new ProgressBar());

        // 2) ARQUITETURA DO MODELO
        // A estrutura é definida em DigitClassifier.Architecture.newBlock(),
        // compartilhada com a inferência. Escolha com -Dmodelo.arquitetura=mlp|cnn
        // (padrão: mlp). Cada arquitetura salva em arquivo próprio.
        DigitClassifier.Architecture architecture = DigitClassifier.Architecture.fromProperty();
        // O Model guarda memória nativa: try-with-resources o fecha mesmo se o treino falhar.
        try (Model model = Model.newInstance(architecture.modelName())) {
            model.setBlock(architecture.newBlock());

            // 3) CONFIGURAÇÃO DO TREINO
            // - Loss (perda): softmaxCrossEntropyLoss, a escolha padrão para
            //   classificação com várias classes. Mede o quão "errada" está a
            //   previsão; o treino tenta minimizar esse valor.
            // - Accuracy: métrica de acompanhamento (% de acertos).
            // - Logging: imprime o progresso de cada época no console.
            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .addEvaluator(new Accuracy())
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            // Threads daemon: se o pool ficasse "vivo", o programa nunca terminaria.
            if (threads > 0) {
                config.optExecutorService(java.util.concurrent.Executors.newFixedThreadPool(threads, runnable -> {
                    Thread thread = new Thread(runnable, "carregamento-dados");
                    thread.setDaemon(true);
                    return thread;
                }));
            }

            // Device: por padrão o DJL decide sozinho (todas as GPUs detectadas ou CPU).
            // -Dtreino.device=gpu força a GPU 0 mesmo quando o DJL não a detecta
            // (acontece sem o CUDA Toolkit instalado); -Dtreino.device=cpu força a CPU.
            String deviceOption = System.getProperty("treino.device", "auto");
            if (deviceOption.equals("gpu")) {
                config.optDevices(new Device[] {Device.gpu(0)});
            } else if (deviceOption.equals("cpu")) {
                config.optDevices(new Device[] {Device.cpu()});
            }

            // Informa onde o treino vai rodar. O DJL usa todas as GPUs visíveis e,
            // se não houver nenhuma, a CPU. Para forçar CPU: CUDA_VISIBLE_DEVICES="".
            Device[] devices = config.getDevices();
            System.out.println("Engine: " + Engine.getInstance().getEngineName()
                    + " " + Engine.getInstance().getVersion());
            System.out.println("GPUs disponíveis: " + Engine.getInstance().getGpuCount());
            System.out.println("Treinando em: " + java.util.Arrays.toString(devices)
                    + " | lote=" + batchSize + " | épocas=" + epochs + " | semente=" + seed
                    + " | cache=" + cache + " | threads=" + threads);

            // 4) TREINAMENTO
            // O Trainer guarda recursos nativos, por isso é aberto em
            // try-with-resources (fechado automaticamente ao final).
            try (Trainer trainer = model.newTrainer(config)) {

                // Informa o formato de UMA entrada para o Trainer alocar os pesos:
                // (lote, canal, altura, largura) = (1, 1, 28, 28). Cada amostra do
                // ImageFolderDataset tem formato (1, 28, 28) — canal, altura,
                // largura — e o Mlp achata isso para 784 valores; a CNN usa o formato como está.
                trainer.initialize(new Shape(1, 1, DigitClassifier.IMAGE_HEIGHT, DigitClassifier.IMAGE_WIDTH));

                trainer.setMetrics(new Metrics());

                // Loop de treino: para cada época, percorre os lotes do conjunto
                // de treino (ajustando os pesos) e depois avalia no de validação.
                long start = System.nanoTime();
                EasyTrain.fit(trainer, epochs, trainingSet, validationSet);
                double seconds = (System.nanoTime() - start) / 1e9;

                TrainingResult result = trainer.getTrainingResult();
                System.out.println("Acurácia final de validação: "
                        + result.getValidateEvaluation("Accuracy"));

                // Resumo de desempenho (as imagens de validação também são processadas a cada época).
                long images = trainingSet.size() * (long) epochs;
                System.out.printf("=== DESEMPENHO === device=%s lote=%d épocas=%d%n",
                        devices[0], batchSize, epochs);
                System.out.printf("Tempo total de fit: %.1f s | por época: %.1f s | %.0f imagens/s (treino)%n",
                        seconds, seconds / epochs, images / seconds);

                appendResult(label, architecture, devices[0], batchSize, epochs, seed, cache, threads, result, seconds, images / seconds);
            }

            // 5) SALVAR O MODELO
            // Grava os pesos em build/mnist-model/<modelo>-0005.params (o 0005
            // indica a época). É esse arquivo que DigitClassifier carrega depois.
            model.save(modelDir, architecture.modelName());
            System.out.println("Modelo salvo em: " + modelDir.toAbsolutePath());
        }
    }

    /**
     * Acrescenta uma linha ao CSV de resultados ({@link #RESULTS_FILE}), criando o
     * arquivo com cabeçalho na primeira vez. Permite comparar execuções depois.
     */
    private static void appendResult(String label, DigitClassifier.Architecture architecture, Device device, int batchSize, int epochs,
            int seed, boolean cache, int threads, TrainingResult result, double seconds, double imagesPerSecond)
            throws IOException {
        Files.createDirectories(RESULTS_FILE.getParent());
        if (Files.notExists(RESULTS_FILE)) {
            Files.writeString(RESULTS_FILE, "data_hora,rotulo,arquitetura,device,lote,epocas,semente,cache,threads,"
                    + "acc_treino,acc_validacao,perda_treino,perda_validacao,"
                    + "tempo_total_s,tempo_epoca_s,imagens_por_s\n");
        }
        String line = String.format(java.util.Locale.ROOT,
                "%s,%s,%s,%s,%d,%d,%d,%s,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%.0f%n",
                LocalDateTime.now().truncatedTo(ChronoUnit.SECONDS), label, architecture, device, batchSize,
                epochs, seed, cache, threads, result.getTrainEvaluation("Accuracy"),
                result.getValidateEvaluation("Accuracy"), result.getTrainLoss(),
                result.getValidateLoss(), seconds, seconds / epochs, imagesPerSecond);
        Files.writeString(RESULTS_FILE, line, StandardOpenOption.APPEND);
    }
}
