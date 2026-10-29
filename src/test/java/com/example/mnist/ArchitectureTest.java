package com.example.mnist;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;
import com.example.mnist.DigitClassifier.Architecture;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

class ArchitectureTest {

    @AfterEach
    void limpaPropriedade() {
        System.clearProperty("modelo.arquitetura");
    }

    @ParameterizedTest
    @EnumSource(Architecture.class)
    void saidaTemUmScorePorDigito(Architecture architecture) {
        Block block = architecture.newBlock();
        Shape input = new Shape(2, 1, DigitClassifier.IMAGE_HEIGHT, DigitClassifier.IMAGE_WIDTH);
        try (NDManager manager = NDManager.newBaseManager()) {
            block.initialize(manager, DataType.FLOAT32, input);
            NDArray x = manager.zeros(input);
            NDList out = block.forward(new ParameterStore(manager, false), new NDList(x), false);
            assertEquals(new Shape(2, DigitClassifier.NUM_CLASSES), out.singletonOrThrow().getShape());
        }
    }

    @Test
    void cadaArquiteturaTemSeuProprioArquivoDePesos() {
        assertNotEquals(Architecture.MLP.modelName(), Architecture.CNN.modelName());
    }

    @Test
    void padraoEMlpEPropriedadeEscolheCnn() {
        assertEquals(Architecture.MLP, Architecture.fromProperty());
        System.setProperty("modelo.arquitetura", "cnn");
        assertEquals(Architecture.CNN, Architecture.fromProperty());
    }
}
