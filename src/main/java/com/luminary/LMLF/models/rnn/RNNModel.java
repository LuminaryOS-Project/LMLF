package com.luminary.LMLF.models.rnn;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class RNNModel {

    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    private final double[][] weightsIn;
    private final double[][] weightsOut;

    private final double[] hidden;
    private final double[] output;

    public RNNModel(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        weightsIn = new double[hiddenSize][inputSize];
        weightsOut = new double[outputSize][hiddenSize];
        ThreadLocalRandom random = ThreadLocalRandom.current();
        IntStream.range(0, hiddenSize).forEach(i -> IntStream.range(0, inputSize).forEach(j -> weightsIn[i][j] = random.nextDouble() - 0.5));
        IntStream.range(0, outputSize).forEach(i -> IntStream.range(0, hiddenSize).forEach(j -> weightsOut[i][j] = random.nextDouble() - 0.5));

        hidden = new double[hiddenSize];
        output = new double[outputSize];
    }
    public RNNModel() {
        this(100, 32, 1);
    }

    public double[] forward(double[] input) {
        Arrays.setAll(hidden, i -> IntStream.range(0, inputSize)
                .mapToDouble(j -> weightsIn[i][j] * input[j])
                .sum());
        Arrays.setAll(hidden, i -> tanh(hidden[i]));

        Arrays.setAll(output, i -> IntStream.range(0, hiddenSize)
                .mapToDouble(j -> weightsOut[i][j] * hidden[j])
                .sum());

        Arrays.setAll(output, i -> tanh(output[i]));
        return output;
    }
    public void train(double[][] inputs, double[][] targets, int numEpochs, double learningRate) {
        int numInputs = inputs.length;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            for (int i = 0; i < numInputs; i++) {
                double[] input = inputs[i];
                double[] target = targets[i];
                double[] output = forward(input);

                double[] error = IntStream.range(0, outputSize)
                        .mapToDouble(j -> target[j] - output[j])
                        .toArray();

                double[] outputDeriv = DoubleStream.of(output)
                        .map(x -> 1 - x * x)
                        .toArray();

                double[] hiddenDeriv = DoubleStream.of(hidden)
                        .map(x -> 1 - x * x)
                        .toArray();

                double[] outputError = IntStream.range(0, outputSize)
                        .mapToDouble(j -> error[j] * outputDeriv[j])
                        .toArray();

                double[] hiddenError = IntStream.range(0, hiddenSize)
                        .mapToDouble(j -> IntStream.range(0, outputSize)
                                .mapToDouble(k -> weightsOut[k][j] * outputError[k])
                                .sum() * hiddenDeriv[j])
                        .toArray();

                IntStream.range(0, outputSize).forEach(j -> IntStream.range(0, hiddenSize).forEach(k -> weightsOut[j][k] += learningRate * outputError[j] * hidden[k]));
                IntStream.range(0, hiddenSize).forEach(j -> IntStream.range(0, inputSize).forEach(k -> weightsIn[j][k] += learningRate * hiddenError[j] * input[k]));
            }
        }
    }

    public double[] predict(double[] input) {
        return forward(input);
    }

    private double tanh(double x) {
        return (2.0 / (1.0 + Math.exp(-2.0 * x))) - 1.0;
    }
}
