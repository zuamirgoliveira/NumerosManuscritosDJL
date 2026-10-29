package com.example.mnist;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Encapsula o carregamento do modelo MNIST treinado e a execução da
 * inferência sobre uma imagem.
 *
 * <p>Esta classe é o "ponto único de verdade" sobre o modelo: aqui ficam as
 * constantes (tamanho da imagem, número de classes, nome do modelo) e a
 * definição da arquitetura da rede ({@link Architecture#newBlock()}). Tanto o treino
 * ({@link TrainMnist}) quanto a inferência usam esses mesmos valores, o que
 * evita o erro clássico de treinar com uma arquitetura e tentar carregar os
 * pesos em outra.
 *
 * <p>Também é reaproveitada pela versão de linha de comando
 * ({@link InferMnist}) e pela interface gráfica JavaFX
 * ({@code com.example.mnist.ui.InferenceApp}), evitando duplicar a lógica de
 * tradução imagem → tensor → probabilidades.
 *
 * <p>Uso típico (o try-with-resources libera os recursos nativos):
 * <pre>{@code
 * try (DigitClassifier classifier = new DigitClassifier()) {
 *     Classifications result = classifier.classify(Path.of("digito.png"));
 *     System.out.println(result.best().getClassName());
 * }
 * }</pre>
 */
public final class DigitClassifier implements AutoCloseable {

    /** Altura, em pixels, das imagens esperadas pelo modelo (MNIST = 28). */
    public static final int IMAGE_HEIGHT = 28;

    /** Largura, em pixels, das imagens esperadas pelo modelo (MNIST = 28). */
    public static final int IMAGE_WIDTH = 28;

    /** Número de classes de saída: um para cada dígito de 0 a 9. */
    public static final int NUM_CLASSES = 10;

    /** Pasta (relativa ao diretório de execução) onde o modelo é salvo/lido. */
    public static final Path DEFAULT_MODEL_DIR = Path.of("build/mnist-model");

    /** Neurônios das duas camadas ocultas do MLP: 128 e depois 64. */
    private static final int[] HIDDEN_LAYERS = {128, 64};

    /**
     * Arquiteturas disponíveis. Cada uma tem seu próprio nome de modelo, e por
     * isso seu próprio arquivo de pesos (ex.: "mnist-mlp-0005.params",
     * "mnist-cnn-0005.params", onde 0005 é a época do treino), o que permite
     * treinar e comparar as duas sem uma sobrescrever a outra.
     *
     * <p>É a "fonte única de verdade" da estrutura da rede: treino e inferência
     * chamam {@link #newBlock()} para nunca divergirem.
     */
    public enum Architecture {
        /** Rede densa (Multi-Layer Perceptron): 784 → 128 → 64 → 10. */
        MLP("mnist-mlp"),
        /** Rede convolucional: 2 convoluções + pooling + camada densa. */
        CNN("mnist-cnn");

        private final String modelName;

        Architecture(String modelName) {
            this.modelName = modelName;
        }

        /** Nome base do modelo; o DJL o usa para montar o nome do arquivo de pesos. */
        public String modelName() {
            return modelName;
        }

        /**
         * Cria a rede. Entrada: tensor (lote, 1, 28, 28). Saída: 10 valores (um
         * "score" por dígito). A função softmax, que converte os scores em
         * probabilidades, é aplicada depois, no Translator (na inferência) ou
         * dentro da função de perda (no treino).
         */
        public Block newBlock() {
            if (this == MLP) {
                // O Mlp achata os 28 x 28 = 784 pixels por conta própria.
                return new Mlp(IMAGE_HEIGHT * IMAGE_WIDTH, NUM_CLASSES, HIDDEN_LAYERS);
            }
            // As convoluções olham para vizinhanças de pixels (traços, curvas),
            // o que o MLP, que trata cada pixel isolado, não consegue.
            return new SequentialBlock()
                    .add(Conv2d.builder().setKernelShape(new Shape(3, 3)).setFilters(32).build())
                    .add(Activation::relu)
                    .add(Conv2d.builder().setKernelShape(new Shape(3, 3)).setFilters(64).build())
                    .add(Activation::relu)
                    .add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)))
                    .add(Blocks.batchFlattenBlock())
                    .add(Linear.builder().setUnits(128).build())
                    .add(Activation::relu)
                    .add(Linear.builder().setUnits(NUM_CLASSES).build());
        }

        /**
         * Arquitetura escolhida por {@code -Dmodelo.arquitetura=mlp|cnn}; sem a
         * propriedade, {@link #MLP}.
         */
        public static Architecture fromProperty() {
            return valueOf(System.getProperty("modelo.arquitetura", "mlp").toUpperCase(Locale.ROOT));
        }
    }

    private final Model model;
    private final Predictor<Image, Classifications> predictor;

    /**
     * Carrega o modelo da pasta padrão ({@code build/mnist-model}), com a
     * arquitetura indicada por {@link Architecture#fromProperty()}.
     */
    public DigitClassifier() throws IOException, MalformedModelException {
        this(DEFAULT_MODEL_DIR, Architecture.fromProperty());
    }

    /**
     * Carrega o modelo a partir de uma pasta específica.
     *
     * @param modelDir pasta onde o {@link TrainMnist} salvou o arquivo .params
     * @throws IOException se a pasta ou o arquivo de pesos não existirem
     * @throws MalformedModelException se os pesos não combinam com a arquitetura
     */
    public DigitClassifier(Path modelDir) throws IOException, MalformedModelException {
        this(modelDir, Architecture.fromProperty());
    }

    /**
     * Carrega o modelo de uma arquitetura específica a partir de uma pasta.
     *
     * @param modelDir pasta onde o {@link TrainMnist} salvou o arquivo .params
     * @param architecture arquitetura com a qual o modelo foi treinado
     */
    public DigitClassifier(Path modelDir, Architecture architecture)
            throws IOException, MalformedModelException {
        model = Model.newInstance(architecture.modelName());

        // Recria a mesma arquitetura usada no treino ANTES de carregar os
        // pesos: o arquivo .params guarda só os números (pesos), não a
        // estrutura da rede.
        model.setBlock(architecture.newBlock());

        // Procura em modelDir o arquivo de pesos "<modelo>-XXXX.params"
        // (por padrão, o de maior número de época) e preenche a rede com ele.
        model.load(modelDir, architecture.modelName());

        // O Predictor é o objeto que executa a inferência. Ele recebe um
        // Translator que sabe converter Image -> tensor e tensor -> resultado.
        predictor = model.newPredictor(buildTranslator());
    }

    /**
     * Classifica uma imagem de dígito manuscrito.
     *
     * @param imagePath caminho de um arquivo de imagem (PNG, JPG, BMP...)
     * @return as probabilidades de cada dígito (0-9); use {@code best()} para
     *         obter o mais provável
     */
    public Classifications classify(Path imagePath) throws IOException, TranslateException {
        Image image = ImageFactory.getInstance().fromFile(imagePath);
        return predictor.predict(image);
    }

    /**
     * Monta o Translator: a "ponte" entre o mundo da aplicação (uma Image) e
     * o mundo da rede neural (tensores).
     */
    private static Translator<Image, Classifications> buildTranslator() {
        return new Translator<Image, Classifications>() {

            /** PRÉ-PROCESSAMENTO: transforma a imagem no tensor que a rede espera. */
            @Override
            public NDList processInput(TranslatorContext ctx, Image input) {
                // Converte para escala de cinza (1 canal) — o MNIST não tem cor.
                NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.GRAYSCALE);

                // Redimensiona para 28x28, não importa o tamanho original.
                array = NDImageUtils.resize(array, IMAGE_WIDTH, IMAGE_HEIGHT);

                // toTensor: reorganiza (altura, largura, canal) -> (canal,
                // altura, largura) e normaliza os pixels de 0-255 para 0-1.
                // É exatamente o mesmo tratamento do ImageFolderDataset usado
                // no treino (consistência treino/inferência).
                return new NDList(NDImageUtils.toTensor(array));
            }

            /** PÓS-PROCESSAMENTO: transforma a saída da rede em probabilidades. */
            @Override
            public Classifications processOutput(TranslatorContext ctx, NDList list) {
                // A rede devolve 10 "scores" brutos; o softmax os converte em
                // probabilidades que somam 1 (100%).
                NDArray probabilities = list.singletonOrThrow().softmax(0);

                // Nomes das classes: "0", "1", ..., "9".
                List<String> classNames = IntStream.range(0, NUM_CLASSES)
                        .mapToObj(String::valueOf)
                        .collect(Collectors.toList());

                return new Classifications(classNames, probabilities);
            }

            /**
             * A rede trabalha com lotes (batches). STACK empilha as entradas
             * num lote; aqui, com uma única imagem, vira um lote de tamanho 1.
             */
            @Override
            public Batchifier getBatchifier() {
                return Batchifier.STACK;
            }
        };
    }

    /** Libera os recursos nativos (memória do engine PyTorch) do predictor e do modelo. */
    @Override
    public void close() {
        predictor.close();
        model.close();
    }
}
