package com.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.zip.GZIPInputStream;

import java.io.IOException;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

public class MnistDigitApp {

    private static final String BASE_URL = "https://raw.githubusercontent.com/sunsided/mnist/master/";
    private static final String[] FILE_NAMES = {
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
    };

    public static void main(String[] args) throws Exception {
        String dataDir = "./data/mnist-digits";
        String trainImagesPath = dataDir + "/train-images-idx3-ubyte.gz";
        String trainLabelsPath = dataDir + "/train-labels-idx1-ubyte.gz";
        String testImagesPath = dataDir + "/t10k-images-idx3-ubyte.gz";
        String testLabelsPath = dataDir + "/t10k-labels-idx1-ubyte.gz";

        // 1. Download Files with Progress Bar
        downloadFiles(dataDir);

        // 2. Load and Decompress MNIST Data with Progress Tracking
        System.out.println("\n--- Loading Datasets ---");
        DataSetIterator trainIterator = createDataSetIterator(
                trainImagesPath,
                trainLabelsPath, 512);

        DataSetIterator testIterator = createDataSetIterator(
                testImagesPath,
                testLabelsPath, 512);

        // 3. Build the Network Configuration
        System.out.println("\n--- Initializing DL4J Model ---");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(28 * 28)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(128)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // 4. Train Model with Epoch Progress, Loss, and Timing Metrics
        System.out.println("\n--- Training Model ---");
        int epochs = 5;
        for (int i = 1; i <= epochs; i++) {
            long startTime = System.currentTimeMillis();
            trainIterator.reset();

            double totalLoss = 0.0;
            int batchCount = 0;

            while (trainIterator.hasNext()) {
                DataSet batch = trainIterator.next();
                model.fit(batch);
                totalLoss += model.score();
                batchCount++;
            }

            long endTime = System.currentTimeMillis();
            double avgLoss = totalLoss / batchCount;
            long timeTaken = endTime - startTime;

            System.out.printf("Epoch %d/%d - Loss: %.4f - Time Taken: %d ms%n", i, epochs, avgLoss, timeTaken);
        }

        // 5. Evaluate and Print Confusion Matrix
        System.out.println("\n--- Evaluating Model ---");
        Evaluation eval = new Evaluation(10);
        testIterator.reset();

        while (testIterator.hasNext()) {
            DataSet next = testIterator.next();
            INDArray output = model.output(next.getFeatures());
            eval.eval(next.getLabels(), output);
        }

        // Print full statistics alongside the confusion matrix layout
        System.out.println(eval.stats());

        // 6. Show a window with one sample image per class, predicted label and per-class accuracy
        String[] labels = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
        double[] accuracy = perClassAccuracy(eval);
        int[] sampleImage = findOneSamplePerClass(testImagesPath, testLabelsPath);
        showResultsWindow(labels, accuracy, sampleImage, testImagesPath);
    }

    private static double[] perClassAccuracy(Evaluation eval) {
        double[] accuracy = new double[10];
        for (int c = 0; c < 10; c++) {
            int tp = (int) Math.round(eval.getTruePositives().getCount(c));
            int fn = (int) Math.round(eval.getFalseNegatives().getCount(c));
            int total = tp + fn;
            accuracy[c] = (total > 0) ? (double) tp / total : 0.0;
        }
        return accuracy;
    }

    private static int[] findOneSamplePerClass(String imagesPath, String labelsPath) throws IOException {
        int[] result = new int[10];
        java.util.Arrays.fill(result, -1);
        try (DataInputStream imgIn = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)));
             DataInputStream lblIn = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath)))) {

            imgIn.readInt();
            int numImages = imgIn.readInt();
            int numRows = imgIn.readInt();
            int numCols = imgIn.readInt();
            lblIn.readInt();
            lblIn.readInt();

            int featureSize = numRows * numCols;
            byte[] pixel = new byte[featureSize];
            int found = 0;
            for (int i = 0; i < numImages && found < 10; i++) {
                imgIn.readFully(pixel);
                int labelIdx = lblIn.readUnsignedByte();
                if (result[labelIdx] == -1) {
                    result[labelIdx] = i;
                    found++;
                }
            }
        }
        return result;
    }

    private static void showResultsWindow(String[] labels, double[] accuracy, int[] sampleImage, String imagesPath) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("MNIST Digits - Per-Class Accuracy");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            double overall = accuracy[0];
            for (int i = 1; i < accuracy.length; i++) overall = (overall * i + accuracy[i]) / (i + 1);
            JLabel title = new JLabel(String.format("Overall Accuracy: %.2f%%  (%d classes)", overall * 100, labels.length));
            title.setHorizontalAlignment(JLabel.CENTER);
            title.setBorder(BorderFactory.createEmptyBorder(12, 0, 6, 0));
            title.setFont(title.getFont().deriveFont(16.0f));

            JPanel grid = new JPanel(new GridLayout(0, 5, 8, 8));
            BufferedImage[] images = null;
            try {
                images = loadTestImages(imagesPath, sampleImage);
            } catch (IOException e) {
                System.err.println("Failed to load images: " + e.getMessage());
            }

            for (int c = 0; c < 10; c++) {
                JPanel cell = new JPanel();
                cell.setLayout(new BoxLayout(cell, BoxLayout.Y_AXIS));
                cell.setBorder(BorderFactory.createLineBorder(new java.awt.Color(80, 90, 110)));

                BufferedImage img = (images != null && images[c] != null) ? images[c] : null;
                JPanel canvas = new JPanel() {
                    @Override
                    public Dimension getPreferredSize() {
                        return new Dimension(92, 92);
                    }
                    @Override
                    protected void paintComponent(Graphics g) {
                        super.paintComponent(g);
                        if (img != null) {
                            g.fillRect(0, 0, getWidth(), getHeight());
                            g.drawImage(img, 4, 4, getWidth() - 8, getHeight() - 8, this);
                        } else {
                            g.drawString("?", getWidth() / 2 - 5, getHeight() / 2);
                        }
                    }
                };
                canvas.setPreferredSize(new Dimension(92, 92));
                cell.add(canvas);

                JLabel name = new JLabel(labels[c], JLabel.CENTER);
                name.setPreferredSize(new Dimension(150, 20));
                cell.add(name);

                JLabel acc = new JLabel(String.format("%.1f%%", accuracy[c] * 100), JLabel.CENTER);
                acc.setPreferredSize(new Dimension(150, 20));
                cell.add(acc);

                grid.add(cell);
            }

            frame.add(title, BorderLayout.NORTH);
            frame.add(grid, BorderLayout.CENTER);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }

    private static BufferedImage[] loadTestImages(String imagesPath, int[] sampleImage) throws IOException {
        try (DataInputStream imgIn = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)))) {
            imgIn.readInt();
            int numImages = imgIn.readInt();
            int numRows = imgIn.readInt();
            int numCols = imgIn.readInt();

            int featureSize = numRows * numCols;
            byte[] pixel = new byte[featureSize];
            BufferedImage[] out = new BufferedImage[10];

            for (int i = 0; i < numImages; i++) {
                imgIn.readFully(pixel);
                for (int c = 0; c < 10; c++) {
                    if (sampleImage[c] == i) {
                        BufferedImage bi = new BufferedImage(numCols, numRows, BufferedImage.TYPE_INT_ARGB);
                        int p = 0;
                        for (int y = 0; y < numRows; y++) {
                            for (int x = 0; x < numCols; x++) {
                                int b = pixel[p++] & 0xff;
                                int rgb = (255 << 24) | (b << 16) | (b << 8) | b;
                                bi.setRGB(x, y, rgb);
                            }
                        }
                        out[c] = bi;
                    }
                }
            }
            return out;
        }
    }

    private static void downloadFiles(String destinationDir) {
        File dir = new File(destinationDir);
        if (!dir.exists()) dir.mkdirs();

        System.out.println("--- Downloading Dataset ---");
        for (String fileName : FILE_NAMES) {
            File targetFile = new File(dir, fileName);
            if (targetFile.exists()) {
                System.out.println("[Skipped] Already exists: " + fileName);
                continue;
            }

            try {
                System.out.println("downloading: " + BASE_URL + fileName);
                HttpURLConnection connection = (HttpURLConnection) new URL(BASE_URL + fileName).openConnection();
                connection.setRequestProperty("User-Agent", "Mozilla/5.0");
                connection.setInstanceFollowRedirects(true);
                int responseCode = connection.getResponseCode();
                if (responseCode != HttpURLConnection.HTTP_OK) {
                    throw new IOException("HTTP " + responseCode + " for " + fileName);
                }
                String contentType = connection.getContentType();
                if (contentType != null && contentType.startsWith("text/html")) {
                    throw new IOException("Got an HTML page (size " + connection.getContentLength() +
                            "), expected binary data for " + fileName + ". Check the download URL.");
                }
                int fileSize = connection.getContentLength();

                try (InputStream in = connection.getInputStream();
                     FileOutputStream fos = new FileOutputStream(targetFile)) {

                    byte[] buffer = new byte[4096];
                    int bytesRead;
                    int totalBytesRead = 0;

                    while ((bytesRead = in.read(buffer)) != -1) {
                        fos.write(buffer, 0, bytesRead);
                        totalBytesRead += bytesRead;

                        if (fileSize > 0) {
                            int progress = (int) (((double) totalBytesRead / fileSize) * 100);
                            System.out.print("\rDownloading " + fileName + ": [" + progress + "%]");
                        }
                    }
                    System.out.println(" -> Saved successfully.");
                }
            } catch (Exception e) {
                System.err.println("\nFailed to download " + fileName + ": " + e.getMessage());
            }
        }
    }

    private static DataSetIterator createDataSetIterator(String imagesPath, String labelsPath, int batchSize) throws IOException {
        try (DataInputStream imgIn = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesPath)));
             DataInputStream lblIn = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsPath)))) {

            imgIn.readInt(); // Magic number
            int numImages = imgIn.readInt();
            int numRows = imgIn.readInt();
            int numCols = imgIn.readInt();

            lblIn.readInt(); // Magic number
            int numLabels = lblIn.readInt();

            if (numImages != numLabels) {
                throw new IllegalArgumentException("Image count does not match Label count!");
            }

            int featureSize = numRows * numCols;
            float[] allFeatures = new float[numImages * featureSize];
            float[] allLabels = new float[numImages * 10];

            String targetName = new File(imagesPath).getName();
            int progressInterval = numImages / 10;

            for (int i = 0; i < numImages; i++) {
                int featureOffset = i * featureSize;
                for (int j = 0; j < featureSize; j++) {
                    allFeatures[featureOffset + j] = (imgIn.readUnsignedByte()) / 255.0f;
                }

                int labelIdx = lblIn.readUnsignedByte();
                allLabels[i * 10 + labelIdx] = 1.0f;

                if (i % progressInterval == 0 || i == numImages - 1) {
                    int percent = (int) (((double) (i + 1) / numImages) * 100);
                    System.out.print("\rLoading " + targetName + ": [" + percent + "%]");
                }
            }
            System.out.println();

            INDArray features = Nd4j.create(allFeatures, new int[]{numImages, featureSize});
            INDArray labels = Nd4j.create(allLabels, new int[]{numImages, 10});
            return new BulkDataSetIterator(features, labels, batchSize);
        }
    }

    private static class BulkDataSetIterator implements DataSetIterator {
        private final INDArray features;
        private final INDArray labels;
        private final int batchSize;
        private final int totalSamples;
        private int cursor = 0;

        public BulkDataSetIterator(INDArray features, INDArray labels, int batchSize) {
            this.features = features;
            this.labels = labels;
            this.batchSize = batchSize;
            this.totalSamples = features.rows();
        }

        @Override
        public boolean hasNext() {
            return cursor < totalSamples;
        }

        @Override
        public DataSet next() {
            return next(batchSize);
        }

        @Override
        public DataSet next(int num) {
            if (!hasNext()) throw new NoSuchElementException();
            int end = Math.min(cursor + num, totalSamples);
            INDArray featureSlice = features.get(NDArrayIndex.interval(cursor, end), NDArrayIndex.all());
            INDArray labelSlice = labels.get(NDArrayIndex.interval(cursor, end), NDArrayIndex.all());
            cursor = end;
            return new DataSet(featureSlice, labelSlice);
        }

        @Override
        public int inputColumns() { return 28 * 28; }
        @Override
        public int totalOutcomes() { return 10; }
        @Override
        public boolean resetSupported() { return true; }
        @Override
        public boolean asyncSupported() { return false; }
        @Override
        public void reset() { cursor = 0; }
        @Override
        public int batch() { return batchSize; }
        @Override
        public void setPreProcessor(DataSetPreProcessor preProcessor) {}
        @Override
        public DataSetPreProcessor getPreProcessor() { return null; }
        @Override
        public List<String> getLabels() { return null; }
    }
}
