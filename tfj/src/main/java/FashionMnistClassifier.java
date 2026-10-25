import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Result;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.core.Assign;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;

// Multidimensional layout wrappers
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.zip.GZIPInputStream;
import java.util.Arrays;

public class FashionMnistClassifier {

    private static final String DATA_DIR = FashionMnistDownloader.DEFAULT_DIRECTORY + "/";
    private static final int IMAGE_SIZE = 28;
    private static final int NUM_CLASSES = 10;
    
    // Fashion MNIST Class Labels map (matching DL4J formatting standards)
    private static final String[] LABELS = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    public static void main(String[] args) throws Exception {
        FashionMnistDownloader.ensureAvailable(DATA_DIR, false);
        System.out.println("=================================================================");
        System.out.println("   TensorFlow Java Engine: Initializing Fashion MNIST Pipeline   ");
        System.out.println("=================================================================");
        
        // 1. Load and parse raw datasets (Train & Test)
        byte[] rawTrainImages = readGzipFile(DATA_DIR + "train-images-idx3-ubyte.gz");
        byte[] rawTrainLabels = readGzipFile(DATA_DIR + "train-labels-idx1-ubyte.gz");
        byte[] rawTestImages = readGzipFile(DATA_DIR + "t10k-images-idx3-ubyte.gz");
        byte[] rawTestLabels = readGzipFile(DATA_DIR + "t10k-labels-idx1-ubyte.gz");

        int numTrainImages = ByteBuffer.wrap(rawTrainImages, 4, 4).getInt();
        int numTestImages = ByteBuffer.wrap(rawTestImages, 4, 4).getInt();

        float[][][] trainImages = parseImages(rawTrainImages, numTrainImages);
        long[] trainLabels = parseLabels(rawTrainLabels, numTrainImages);
        float[][][] testImages = parseImages(rawTestImages, numTestImages);
        long[] testLabels = parseLabels(rawTestLabels, numTestImages);

        // 2. Build Computational Graph
        try (Graph g = new Graph(); Session session = new Session(g)) {
            Ops tf = Ops.create(g);

            // Shape inputs: [BatchSize, 28, 28]
            Placeholder<TFloat32> inputPlaceholder = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, IMAGE_SIZE, IMAGE_SIZE)));
            Placeholder<TInt64> labelsPlaceholder = tf.placeholder(TInt64.class, Placeholder.shape(Shape.of(-1)));

            // Layer Transformations (Flatten)
            Operand<TFloat32> flatten = tf.reshape(inputPlaceholder, tf.constant(new long[]{-1, IMAGE_SIZE * IMAGE_SIZE}));

            // Weights and Biases Configurations
            Variable<TFloat32> w1 = tf.variable(Shape.of(784, 128), TFloat32.class);
            Variable<TFloat32> b1 = tf.variable(Shape.of(128), TFloat32.class);
            Variable<TFloat32> w2 = tf.variable(Shape.of(128, NUM_CLASSES), TFloat32.class);
            Variable<TFloat32> b2 = tf.variable(Shape.of(NUM_CLASSES), TFloat32.class);

            // Initialize structural variables
            session.runner().addTarget(tf.assign(w1, tf.random.statelessRandomNormal(tf.constant(new long[]{784, 128}), tf.constant(new long[]{1, 2})))).run();
            session.runner().addTarget(tf.assign(b1, tf.zeros(tf.constant(new long[]{128}), TFloat32.class))).run();
            session.runner().addTarget(tf.assign(w2, tf.random.statelessRandomNormal(tf.constant(new long[]{128, NUM_CLASSES}), tf.constant(new long[]{3, 4})))).run();
            session.runner().addTarget(tf.assign(b2, tf.zeros(tf.constant(new long[]{NUM_CLASSES}), TFloat32.class))).run();

            // Forward Architecture Execution Block
            Operand<TFloat32> layer1 = tf.nn.relu(tf.math.add(tf.linalg.matMul(flatten, w1), b1));
            Operand<TFloat32> logits = tf.math.add(tf.linalg.matMul(layer1, w2), b2);
            Operand<TInt64> predictedLabels = tf.math.argMax(tf.nn.softmax(logits), tf.constant(1), TInt64.class);

            // Loss Calculation Node (Cross-Entropy Approximated Implementation)
            //Operand<TFloat32> loss = tf.math.reduceMean(tf.nn.softmaxCrossEntropyWithLogits(logits, tf.oneHot(labelsPlaceholder, tf.constant(NUM_CLASSES), tf.constant(1.0f), tf.constant(0.0f))).loss(), tf.constant(0));

// Loss Calculation Node (Cross-Entropy Approximated Implementation via tf.math.mean)
Operand<TFloat32> loss = tf.math.mean(
    tf.nn.softmaxCrossEntropyWithLogits(
        logits, 
        tf.oneHot(labelsPlaceholder, tf.constant(NUM_CLASSES), tf.constant(1.0f), tf.constant(0.0f))
    ).loss(), 
    tf.constant(0)
);

            // Basic Gradient Descent Optimization Step Configuration (Learning Rate: 0.01)
           /* org.tensorflow.op.core.Gradients gradients = g.gradients(loss, Arrays.asList(w1, b1, w2, b2));
            Operand<TFloat32> updateW1 = tf.assignSub(w1, tf.math.mul(gradients.dy(0), tf.constant(0.01f)));
            Operand<TFloat32> updateB1 = tf.assignSub(b1, tf.math.mul(gradients.dy(1), tf.constant(0.01f)));
            Operand<TFloat32> updateW2 = tf.assignSub(w2, tf.math.mul(gradients.dy(2), tf.constant(0.01f)));
            Operand<TFloat32> updateB2 = tf.assignSub(b2, tf.math.mul(gradients.dy(3), tf.constant(0.01f)));
*/

// Correctly compile structural backpropagation tree via Gradients.create()
org.tensorflow.op.core.Gradients gradients = org.tensorflow.op.core.Gradients.create(
    tf.scope(), 
    loss, 
    Arrays.asList(w1, b1, w2, b2)
);

// Apply classic Gradient Descent parameter updates
Operand<TFloat32> updateW1 = tf.assignSub(w1, tf.math.mul(gradients.dy(0), tf.constant(0.01f)));
Operand<TFloat32> updateB1 = tf.assignSub(b1, tf.math.mul(gradients.dy(1), tf.constant(0.01f)));
Operand<TFloat32> updateW2 = tf.assignSub(w2, tf.math.mul(gradients.dy(2), tf.constant(0.01f)));
Operand<TFloat32> updateB2 = tf.assignSub(b2, tf.math.mul(gradients.dy(3), tf.constant(0.01f)));



            System.out.println("\nExecuting 5 Epochs Training Loop...");
            System.out.println("-----------------------------------------------------------------");

            try (TFloat32 inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(trainImages));
                 TInt64 labelTensor = TInt64.tensorOf(StdArrays.ndCopyOf(trainLabels))) {

                for (int epoch = 1; epoch <= 5; epoch++) {
                    long startTime = System.currentTimeMillis();

                    // Run model optimize logic via native pipeline
                    try (Result trainResult = session.runner()
                            .feed(inputPlaceholder.asOutput(), inputTensor)
                            .feed(labelsPlaceholder.asOutput(), labelTensor)
                            .fetch(loss)
                            .addTarget(updateW1)
                            .addTarget(updateB1)
                            .addTarget(updateW2)
                            .addTarget(updateB2)
                            .run()) {

                        TFloat32 calculatedLoss = (TFloat32) trainResult.get(0);
                        long endTime = System.currentTimeMillis();
                        
                        System.out.printf("Epoch %d/5 Finished | Loss: %.5f | Duration: %d ms%n", 
                                epoch, calculatedLoss.getFloat(), (endTime - startTime));
                    }
                }
            }

            // 3. Evaluation and DL4J-Style Confusion Matrix Processing
            System.out.println("\n-----------------------------------------------------------------");
            System.out.println("Evaluating Model and Printing Confusion Matrix...");
            System.out.println("-----------------------------------------------------------------");

            int[][] confusionMatrix = new int[NUM_CLASSES][NUM_CLASSES];

            try (TFloat32 testInputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(testImages))) {
                try (Result evalResult = session.runner()
                        .feed(inputPlaceholder.asOutput(), testInputTensor)
                        .fetch(predictedLabels)
                        .run()) {

                    TInt64 testPredictions = (TInt64) evalResult.get(0);
                    
                    // Increment coordinates based on (Actual, Predicted)
                    for (int i = 0; i < numTestImages; i++) {
                        int actualClass = (int) testLabels[i];
                        int predictedClass = (int) testPredictions.getLong(i);
                        confusionMatrix[actualClass][predictedClass]++;
                    }
                }
            }

            // Render output matrix to terminal
            printConfusionMatrix(confusionMatrix);
        }
    }

    // High performance terminal alignment formatting
    private static void printConfusionMatrix(int[][] cm) {
        System.out.println("\n=== Confusion Matrix (Actual Row vs Predicted Column) ===");
        System.out.printf("%-15s", "Actual \\ Pred");
        for (int i = 0; i < NUM_CLASSES; i++) {
            System.out.printf(" [%d]  ", i);
        }
        System.out.println();

        int correctPredictions = 0;
        int totalElements = 0;

        for (int r = 0; r < NUM_CLASSES; r++) {
            System.out.printf("%-15s", String.format("[%d] %s", r, LABELS[r]));
            for (int c = 0; c < NUM_CLASSES; c++) {
                System.out.printf("%5d ", cm[r][c]);
                if (r == c) correctPredictions += cm[r][c];
                totalElements += cm[r][c];
            }
            System.out.println();
        }

        double accuracy = ((double) correctPredictions / totalElements) * 100.0;
        System.out.println("=========================================================");
        System.out.printf("Evaluation Done! Overall Testing Accuracy: %.2f%%%n", accuracy);
        System.out.println("=========================================================");
    }

    private static float[][][] parseImages(byte[] raw, int total) {
        float[][][] images = new float[total][IMAGE_SIZE][IMAGE_SIZE];
        int offset = 16;
        for (int i = 0; i < total; i++) {
            for (int r = 0; r < IMAGE_SIZE; r++) {
                for (int c = 0; c < IMAGE_SIZE; c++) {
                    images[i][r][c] = (raw[offset++] & 0xFF) / 255.0f;
                }
            }
        }
        return images;
    }

    private static long[] parseLabels(byte[] raw, int total) {
        long[] labels = new long[total];
        int offset = 8;
        for (int i = 0; i < total; i++) {
            labels[i] = raw[offset++] & 0xFF;
        }
        return labels;
    }

private static byte[] readGzipFile(String path) throws IOException {
	File file = new File(path);
	if (!file.exists()) {
		throw new FileNotFoundException("Missing dataset asset path: " + file.getAbsolutePath());
		}
		try (GZIPInputStream gis = new GZIPInputStream(new FileInputStream(file));
		ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
			byte[] buffer = new byte[8192];
		int len;
while ((len = gis.read(buffer)) > 0) 
{
	baos.write(buffer, 0, len);
}
return baos.toByteArray();

}

}

}