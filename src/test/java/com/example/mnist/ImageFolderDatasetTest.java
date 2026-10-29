package com.example.mnist;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.Record;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ImageFolderDatasetTest {

    @TempDir
    Path root;

    @BeforeEach
    void createImages() throws IOException {
        TestImages.write(root.resolve("0"), "a.png", 1);
        TestImages.write(root.resolve("0"), "b.png", 2);
        TestImages.write(root.resolve("1"), "a.png", 3);
        // Devem ser ignorados: pasta de nome não numérico e arquivo que não é .png.
        TestImages.write(root.resolve("abc"), "x.png", 4);
        Files.writeString(root.resolve("0").resolve("leiame.txt"), "ignorar");
    }

    private ImageFolderDataset dataset(boolean cache) throws IOException {
        ImageFolderDataset ds = ImageFolderDataset.builder()
                .setRoot(root)
                .setSampling(2, false)
                .setCacheInMemory(cache)
                .build();
        ds.prepare(null);
        return ds;
    }

    @Test
    void indexaSomentePngsDePastasNumericas() throws IOException {
        assertEquals(3, dataset(false).availableSize());
    }

    @Test
    void nomeDaPastaEhORotuloEImagensSaemOrdenadas() throws Exception {
        ImageFolderDataset ds = dataset(false);
        try (NDManager manager = NDManager.newBaseManager()) {
            assertEquals(0f, ds.get(manager, 0).getLabels().get(0).getFloat());
            assertEquals(0f, ds.get(manager, 1).getLabels().get(0).getFloat());
            assertEquals(1f, ds.get(manager, 2).getLabels().get(0).getFloat());
        }
    }

    @Test
    void tensorTemFormatoCanalAlturaLarguraEValoresEntre0e1() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray data = dataset(false).get(manager, 0).getData().get(0);
            assertEquals(new Shape(1, 28, 28), data.getShape());
            assertTrue(data.min().getFloat() >= 0f);
            assertTrue(data.max().getFloat() <= 1f);
            assertTrue(data.max().getFloat() > 0f);
        }
    }

    @Test
    void cacheEmMemoriaDaOMesmoResultadoQueLerDoDisco() throws Exception {
        ImageFolderDataset semCache = dataset(false);
        ImageFolderDataset comCache = dataset(true);
        try (NDManager manager = NDManager.newBaseManager()) {
            for (long i = 0; i < 3; i++) {
                Record a = semCache.get(manager, i);
                Record b = comCache.get(manager, i);
                assertArrayEquals(a.getData().get(0).toFloatArray(), b.getData().get(0).toFloatArray());
                assertEquals(a.getLabels().get(0).getFloat(), b.getLabels().get(0).getFloat());
            }
        }
    }

    @Test
    void semSubpastasDeClasseLancaErroComOrientacao(@TempDir Path vazio) {
        ImageFolderDataset ds = ImageFolderDataset.builder().setRoot(vazio).setSampling(2, false).build();
        IOException e = assertThrows(IOException.class, () -> ds.prepare(null));
        assertTrue(e.getMessage().contains("PrepareTrainingImages"));
    }
}
