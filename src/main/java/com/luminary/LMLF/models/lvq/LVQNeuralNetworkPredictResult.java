package com.luminary.LMLF.models.lvq;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class LVQNeuralNetworkPredictResult {

    // Distances to different class centers
    private final List<Map.Entry<Double, Integer>> distances;

    /**
     * Create a new predict result. The constructor can be accessed only by a neural network.
     */
    LVQNeuralNetworkPredictResult(TreeMap<Double, Integer> distances) {
        this.distances = new ArrayList<>(distances.entrySet());
    }

    /**
     * Get the best matched category of predicted data.
     *
     * @return the best matched category of predicted data
     */
    public int getCategory() {
        return distances.get(0).getValue();
    }

    /**
     * Get the difference (Euclidean distance) between the input and the best matched category.
     *
     * @return likelihood between the input and the matched category
     */
    public double getDifference() {
        return distances.get(0).getKey();
    }

    /**
     * Get the likelihood between the input and the best matched category.
     *
     * @return the likelihood between the input and the best matched category
     */
    public double getLikelihood() {
        return distances.get(0).getKey() / distances.get(1).getKey();
    }

}
