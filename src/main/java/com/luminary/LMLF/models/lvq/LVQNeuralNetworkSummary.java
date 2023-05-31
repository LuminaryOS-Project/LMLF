package com.luminary.LMLF.models.lvq;

import lombok.Getter;

/**
 * A wrapper containing statistics about a LVQ Neural Network.
 */
public final class LVQNeuralNetworkSummary {
    // times the network has been trained
    @Getter
    private final int epoch;

    // current step_size of the network
    @Getter
    private final double step_size;

    // the number of input vectors and neurons in output layer
    @Getter
    private final int input_count;
    @Getter
    private final int output_count;

    /**
     * Create a summary for a LVQ neural network. The constructor can be accessed only by a neural network.
     *  @param epoch times the network has been trained
     * @param step_size current step_size of the network
     * @param input_count number of input vectors
     * @param output_count number of neurons in output layer
     */
    LVQNeuralNetworkSummary(int epoch, double step_size, int input_count, int output_count) {
        this.epoch = epoch;
        this.step_size = step_size;
        this.input_count = input_count;
        this.output_count = output_count;
    }

}
