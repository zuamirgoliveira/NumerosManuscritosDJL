package com.example.mnist;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.Record;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.util.Progress;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * Dataset customizado que lê imagens diretamente do disco, dispensando o
 * módulo {@code ai.djl:basicdataset}.
 *
 * <p>Espera uma pasta raiz organizada em uma subpasta por classe, onde o
 * NOME da subpasta é o próprio rótulo (padrão "pasta por classe", usado por
 * praticamente todo framework de deep learning):
 *
 * <pre>
 * images-training/
 *   0/  -&gt; imagens do dígito 0
 *   1/  -&gt; imagens do dígito 1
 *   ...
 *   9/  -&gt; imagens do dígito 9
 * </pre>
 *
 * <p>Diferente do Mnist do basicdataset (que carrega tudo de uma vez em um
 * grande array na memória), este dataset é "preguiçoso": {@link #prepare}
 * apenas indexa os caminhos dos arquivos, e cada imagem só é de fato lida e
 * decodificada do disco quando {@link #get} é chamado para aquele índice —
 * normalmente uma vez por época, por imagem, durante o treino.
 *
 * <p>Opcionalmente ({@code setCacheInMemory(true)}), {@link #prepare} também
 * decodifica TODAS as imagens uma única vez e as guarda na memória como bytes
 * (o MNIST inteiro ocupa cerca de 55 MB). As épocas seguintes deixam de tocar o
 * disco e de decodificar PNG, que era o gargalo do treino.
 *
 * <p>Por herdar de {@link RandomAccessDataset}, ganha de graça o
 * embaralhamento e a divisão em lotes (batches) — basta chamar
 * {@code setSampling(batchSize, shuffle)} no builder.
 */
public final class ImageFolderDataset extends RandomAccessDataset {

    /** Pasta raiz que contém uma subpasta por classe. */
    private final Path root;

    // Duas listas "paralelas": imagePaths[i] é o arquivo e imageLabels[i] é o
    // dígito (rótulo) correspondente a ele.
    private final List<Path> imagePaths = new ArrayList<>();
    private final List<Integer> imageLabels = new ArrayList<>();

    /** Se verdadeiro, as imagens são decodificadas uma vez em {@link #prepare} e guardadas aqui. */
    private final boolean cacheInMemory;

    /** Pixels (uint8) e formato (altura, largura, canais) de cada imagem; só usado com cache. */
    private final List<byte[]> cachedPixels = new ArrayList<>();
    private final List<Shape> cachedShapes = new ArrayList<>();

    /** Evita indexar as pastas duas vezes se prepare() for chamado de novo. */
    private boolean prepared;

    private ImageFolderDataset(Builder builder) {
        super(builder);
        this.root = builder.root;
        this.cacheInMemory = builder.cacheInMemory;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Escaneia a pasta raiz uma única vez: para cada subpasta (0-9), lista
     * os arquivos .png e guarda (caminho, rótulo) em memória. As imagens em
     * si NÃO são carregadas aqui — só os caminhos.
     */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        // Só considera subpastas cujo nome é um número (o rótulo). Isso ignora
        // pastas estranhas (ex.: ".git", "tmp") em vez de quebrar no parseInt.
        File[] classDirs = root.toFile().listFiles(
                file -> file.isDirectory() && file.getName().matches("\\d+"));
        if (classDirs == null || classDirs.length == 0) {
            throw new IOException(
                    "Nenhuma subpasta de classe encontrada em " + root.toAbsolutePath()
                            + ". Rode a classe PrepareTrainingImages primeiro.");
        }
        // Ordena numericamente (0, 1, 2, ..., 10) em vez de alfabeticamente.
        Arrays.sort(classDirs, Comparator.comparingInt(dir -> Integer.parseInt(dir.getName())));

        for (File classDir : classDirs) {
            // O nome da pasta é o rótulo (ex.: pasta "7" -> dígito 7).
            int label = Integer.parseInt(classDir.getName());

            File[] files = classDir.listFiles(
                    (dir, name) -> name.toLowerCase(Locale.ROOT).endsWith(".png"));
            if (files == null) {
                continue;
            }
            // listFiles() não garante ordem; ordenar deixa a indexação
            // determinística (mesma ordem em qualquer máquina/execução).
            Arrays.sort(files);

            for (File file : files) {
                imagePaths.add(file.toPath());
                imageLabels.add(label);
            }
        }

        if (cacheInMemory) {
            decodeAllImages();
        }

        prepared = true;

        if (progress != null) {
            progress.reset("Indexando imagens em " + root, imagePaths.size());
            progress.end();
        }
    }

    /**
     * Decodifica todas as imagens (em paralelo) para escala de cinza e guarda os
     * bytes na memória. Usa a mesma conversão do caminho sem cache
     * ({@code toNDArray(GRAYSCALE)}), então o resultado final é idêntico.
     */
    private void decodeAllImages() {
        int n = imagePaths.size();
        byte[][] pixels = new byte[n][];
        Shape[] shapes = new Shape[n];
        java.util.stream.IntStream.range(0, n).parallel().forEach(i -> {
            try (NDManager temp = NDManager.newBaseManager()) {
                Image image = ImageFactory.getInstance().fromFile(imagePaths.get(i));
                NDArray array = image.toNDArray(temp, Image.Flag.GRAYSCALE);
                pixels[i] = array.toByteArray();
                shapes[i] = array.getShape();
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        });
        cachedPixels.addAll(Arrays.asList(pixels));
        cachedShapes.addAll(Arrays.asList(shapes));
    }

    /**
     * Carrega e decodifica UMA imagem do disco (a de índice {@code index}) e a
     * transforma no par (dado, rótulo) que o treinador espera.
     */
    @Override
    public Record get(NDManager manager, long index) throws IOException {
        int i = Math.toIntExact(index);

        NDArray array;
        if (cacheInMemory) {
            // Bytes já decodificados em prepare(): só cria o tensor uint8.
            array = manager.create(java.nio.ByteBuffer.wrap(cachedPixels.get(i)),
                    cachedShapes.get(i), ai.djl.ndarray.types.DataType.UINT8);
        } else {
            // Lê o arquivo PNG do disco (só neste momento — carregamento preguiçoso).
            Image image = ImageFactory.getInstance().fromFile(imagePaths.get(i));
            array = image.toNDArray(manager, Image.Flag.GRAYSCALE);
        }

        // Converte para escala de cinza (1 canal) e depois para um tensor
        // float com valores entre 0 e 1 (formato: canal, altura, largura) — o
        // mesmo tratamento aplicado na inferência por DigitClassifier,
        // garantindo consistência entre treino e inferência.
        NDArray data = NDImageUtils.toTensor(array);

        // O rótulo é um escalar com o número do dígito (0 a 9).
        NDArray label = manager.create((float) imageLabels.get(i));

        // Record = um exemplo de treino: (entradas, rótulos).
        return new Record(new NDList(data), new NDList(label));
    }

    /** Quantidade total de imagens indexadas (válido depois de {@link #prepare}). */
    @Override
    public long availableSize() {
        return imagePaths.size();
    }

    /**
     * Builder no mesmo padrão usado pelos datasets prontos do DJL
     * (ex.: Mnist.Builder). Herda de {@code RandomAccessDataset.BaseBuilder},
     * que já traz opções como {@code setSampling(batchSize, shuffle)}.
     */
    public static final class Builder extends BaseBuilder<Builder> {

        Path root;
        boolean cacheInMemory;

        /** Define a pasta raiz (ex.: "images-training") a ser lida. */
        public Builder setRoot(Path root) {
            this.root = root;
            return this;
        }

        /** Se verdadeiro, decodifica todas as imagens uma vez e as mantém na memória. */
        public Builder setCacheInMemory(boolean cacheInMemory) {
            this.cacheInMemory = cacheInMemory;
            return this;
        }

        /** Exigido pelo padrão "builder genérico" do DJL: devolve o próprio builder. */
        @Override
        protected Builder self() {
            return this;
        }

        public ImageFolderDataset build() {
            return new ImageFolderDataset(this);
        }
    }
}
