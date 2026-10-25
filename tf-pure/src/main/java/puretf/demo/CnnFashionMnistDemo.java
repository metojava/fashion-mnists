package puretf.demo;

import puretf.Optim;
import puretf.chart.EvalCharts;
import puretf.chart.TrainingCurves;
import puretf.data.IdxDataset;
import puretf.data.FashionMnist;
import puretf.nn.*;

/**
 * Fashion-MNIST with a CONVOLUTIONAL network - everything computed by the
 * pure-Java autograd engine (im2col conv, max pooling, Adam).
 *
 * Architecture:
 *   [28x28x1] -> Conv(8,3x3,SAME)+ReLU -> MaxPool(2x2)
 *             -> Conv(16,3x3,SAME)+ReLU -> MaxPool(2x2)
 *             -> Flatten -> Dense(128)+ReLU -> Dropout -> Dense(10)
 *
 * Usage: run.cmd fashion-cnn [epochs=5] [--fast] [--batch=N]
 *   --fast trains on a 12k/2k subset so a first result lands in minutes.
 *   --batch sets the train and evaluation batch size (default 96).
 */
public final class CnnFashionMnistDemo {

    private static final String[] NAMES = {
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    public static void main(String[] args) throws Exception {
        String dir = FashionMnist.DEFAULT_DIRECTORY;
        int epochs = 5;
        boolean fast = false;
        int batch = 512;
        for (String a : args) {
            if (a.matches("\\d+")) epochs = Integer.parseInt(a);
            else if (a.equals("--fast")) fast = true;
            else if (a.startsWith("--batch=")) {
                batch = Integer.parseInt(a.substring("--batch=".length()));
                if (batch <= 0) throw new IllegalArgumentException(
                        "--batch must be positive");
            } else dir = a;
        }

        if (epochs <= 0 || batch <= 0) throw new IllegalArgumentException("epochs and batch must be positive");
        FashionMnist.ensureAvailable(dir, false);
        long t0 = System.currentTimeMillis();
        var trI = IdxDataset.loadImages(java.nio.file.Path.of(dir, "train-images-idx3-ubyte.gz").toString());
        var trL = IdxDataset.loadLabels(java.nio.file.Path.of(dir, "train-labels-idx1-ubyte.gz").toString());
        var teI = IdxDataset.loadImages(java.nio.file.Path.of(dir, "t10k-images-idx3-ubyte.gz").toString());
        var teL = IdxDataset.loadLabels(java.nio.file.Path.of(dir, "t10k-labels-idx1-ubyte.gz").toString());

        float[][] trainX = trI.pixels();
        int[] trainY = trL.labels();
        float[][] testX = teI.pixels();
        int[] testY = teL.labels();
        if (fast) {
            trainX = java.util.Arrays.copyOf(trainX, 12000);
            trainY = java.util.Arrays.copyOf(trainY, 12000);
            testX = java.util.Arrays.copyOf(testX, 2000);
            testY = java.util.Arrays.copyOf(testY, 2000);
        }
        System.out.printf("loaded %d train / %d test 28x28 images (%.1fs)%n",
                trainX.length, testX.length,
                (System.currentTimeMillis() - t0) / 1000.0);

/*	
        Sequential net = new Sequential()
                .add(new Reshape(-1, 28, 28, 1))
                .add(new Conv2D(8, 3, 3, 1, 1, Padding.SAME))
                .add(new ReLU())
                .add(new MaxPool2D(2, 2, 2, 2, Padding.VALID))
                .add(new Conv2D(16, 3, 3, 1, 1, Padding.SAME))
                .add(new ReLU())
                .add(new MaxPool2D(2, 2, 2, 2, Padding.VALID))
                .add(new Flatten())
                .add(new Dense(128, Dense.Act.RELU))
                .add(new Dropout(0.15f))
                .add(new Dense(10));
	*/		
        Sequential net = new Sequential()
			.add(new Reshape(-1, 28, 28))           // input_shape=(28,28)
			.add(new Flatten())                      // Flatten → [B, 784]
			.add(new Dense(128, Dense.Act.RELU))     // Dense(128, relu)
			.add(new Dense(10)); 

        History hist = net.fitHistory(trainX, trainY, 10, epochs, batch,
                new Optim.Adam(0.0015f),
                LrSchedule.cosine(0.0015f, 0.0002f, epochs),
                0f, null, true);

        double[] eval = net.evaluate(testX, testY, 10, batch);
        int[][] conf = new int[10][10];
        float[][] logits = net.predict(testX, batch);
        int[] yPred = new int[testX.length];
        for (int i = 0; i < testX.length; i++) {
            yPred[i] = Metrics.argmax(logits[i]);
            conf[testY[i]][yPred[i]]++;
        }
        System.out.println();
        Metrics.printReport(eval[0], eval[1], conf, NAMES);

        int[] sampleIdx = new int[10];
        double[] accPct = new double[10];
        for (int i = 0; i < testX.length; i++) {
            int c = testY[i];
            if (sampleIdx[c] == 0 && yPred[i] == c) sampleIdx[c] = i;
        }
        for (int i = 0; i < 10; i++)
            if (sampleIdx[i] == 0) sampleIdx[i] = i;
        for (int c = 0; c < 10; c++) {
            int total = 0;
            for (int j = 0; j < 10; j++) total += conf[c][j];
            accPct[c] = total > 0 ? 100.0 * conf[c][c] / total : 0;
        }
        EvalCharts.render("Fashion-MNIST CNN - sample predictions",
                testX, 28, sampleIdx, testY, yPred, NAMES, accPct,
                true, "target\\fashion-cnn-samples.png");
        System.out.println("samples saved to target/fashion-cnn-samples.png");

        net.saveTo(java.nio.file.Path.of("target\\fashion-cnn.bin"));
        TrainingCurves.save("Fashion-MNIST CNN - training curves",
                hist, "target\\fashion-cnn-curves.png");
        System.out.println("\nmodel saved to target/fashion-cnn.bin"
                + "\ncurves saved to target/fashion-cnn-curves.png");
    }

    private CnnFashionMnistDemo() {}
}
