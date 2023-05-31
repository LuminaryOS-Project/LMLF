package com.luminary.LMLF.models.rnn;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

public class PredictionUtils {
    private static final AtomicReference<Double> predictionGrabber = new AtomicReference<>();

    private static final Map<String, Double> predictionMap = new LinkedHashMap<>();


    /**
     * Checks if the given message contains spam
     * @param rnnModel The RNNModel to use
     * @param message The message to check
     * @return True if the message contains passed data (eg spam)
     */
    public static boolean contains(RNNModel rnnModel, String message) {
        int inputSize = 100;
        double[] input = new double[inputSize];

        for (int i = 0; i < inputSize; i++) {
            if (i < message.length()) {
                input[i] = (double) message.charAt(i) / 256.0; // normalize input
            } else {
                input[i] = 0.0;
            }
        }
        double[] output = rnnModel.predict(input);

        predictionGrabber.set(output[0]);

        //If the prediction is greater than 0.85 then it will be considered spam
        return output[0] >= 0.85;
    }

    public static double getPrediction() {
        return predictionGrabber.get();
    }

    public static Map<String, Double> getPredictionMap() {
        return predictionMap;
    }
}
