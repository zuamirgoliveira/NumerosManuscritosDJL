package com.example.mnist;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Utilitário de testes: cria imagens PNG 28x28 em tons de cinza. */
final class TestImages {

    private TestImages() {}

    /** Grava um PNG 28x28 cinza em que o valor de cada pixel é {@code (x + y + seed) % 256}. */
    static Path write(Path dir, String name, int seed) throws IOException {
        Files.createDirectories(dir);
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int v = (x + y + seed) % 256;
                image.getRaster().setSample(x, y, 0, v);
            }
        }
        Path file = dir.resolve(name);
        ImageIO.write(image, "png", file.toFile());
        return file;
    }
}
