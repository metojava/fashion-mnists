import java.io.*;
import java.nio.ByteBuffer;
import java.util.zip.GZIPInputStream;

public class FashionMnistDataset {
    public static final int IMAGE_SIZE = 28;

    public static float[][][] parseImages(String path) throws IOException {
        byte[] raw = readGzipFile(path);
        int total = ByteBuffer.wrap(raw, 4, 4).getInt();
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

    public static long[] parseLabels(String path) throws IOException {
        byte[] raw = readGzipFile(path);
        int total = ByteBuffer.wrap(raw, 4, 4).getInt();
        long[] labels = new long[total];
        int offset = 8; 
        for (int i = 0; i < total; i++) {
            labels[i] = raw[offset++] & 0xFF;
        }
        return labels;
    }

    // ====== ENSURE THESE TWO METHODS ARE VISIBLE AND PUBLIC ======
    public static float[][][] getBatchImages(float[][][] source, int start, int size) {
        float[][][] batch = new float[size][IMAGE_SIZE][IMAGE_SIZE];
        System.arraycopy(source, start, batch, 0, size);
        return batch;
    }

    public static long[] getBatchLabels(long[] source, int start, int size) {
        long[] batch = new long[size];
        System.arraycopy(source, start, batch, 0, size);
        return batch;
    }

    private static byte[] readGzipFile(String path) throws IOException {
        File file = new File(path);
        if (!file.exists()) {
            throw new FileNotFoundException("Missing dataset asset: " + file.getAbsolutePath());
        }
        try (GZIPInputStream gis = new GZIPInputStream(new FileInputStream(file));
             ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int len;
            while ((len = gis.read(buffer)) > 0) {
                baos.write(buffer, 0, len);
            }
            return baos.toByteArray();
        }
    }
}
