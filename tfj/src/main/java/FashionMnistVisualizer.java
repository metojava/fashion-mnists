import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class FashionMnistVisualizer {
    private static final int IMAGE_SIZE = 28;
    private static final String[] LABELS = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    public static void displayVisualResults(float[][][] images, long[] actual, long[] predicted, double accuracy) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Fashion MNIST Test Evaluation Grid");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(900, 800);
            frame.setLayout(new BorderLayout());

            JPanel header = new JPanel();
            header.setBackground(new Color(40, 44, 52));
            JLabel title = new JLabel(String.format("Classification Accuracy Matrix Result: %.2f%%", accuracy));
            title.setForeground(Color.WHITE);
            title.setFont(new Font("SansSerif", Font.BOLD, 16));
            header.add(title);
            frame.add(header, BorderLayout.NORTH);

            JPanel grid = new JPanel(new GridLayout(5, 5, 8, 8));
            grid.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

            // Render top 25 images
            for (int i = 0; i < 25; i++) {
                JPanel card = new JPanel(new BorderLayout());
                card.setBorder(BorderFactory.createLineBorder(Color.GRAY, 1));

                BufferedImage img = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY);
                for (int r = 0; r < IMAGE_SIZE; r++) {
                    for (int c = 0; c < IMAGE_SIZE; c++) {
                        int val = (int) (images[i][r][c] * 255);
                        img.setRGB(c, r, (val << 16) | (val << 8) | val);
                    }
                }

                JLabel imgLabel = new JLabel(new ImageIcon(img.getScaledInstance(90, 90, Image.SCALE_FAST)));
                card.add(imgLabel, BorderLayout.CENTER);

                String text = String.format("<html>Pred: %s<br>True: %s</html>", LABELS[(int)predicted[i]], LABELS[(int)actual[i]]);
                JLabel info = new JLabel(text, JLabel.CENTER);
                info.setOpaque(true);
                info.setFont(new Font("SansSerif", Font.PLAIN, 11));
                
                if (predicted[i] == actual[i]) {
                    info.setBackground(new Color(230, 245, 230));
                    info.setForeground(new Color(0, 100, 0));
                } else {
                    info.setBackground(new Color(255, 230, 230));
                    info.setForeground(new Color(150, 0, 0));
                }

                card.add(info, BorderLayout.SOUTH);
                grid.add(card);
            }
            frame.add(new JScrollPane(grid), BorderLayout.CENTER);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
