package com.luminary.LMLF.models.rnn;

import lombok.Getter;

import java.util.HashMap;

public class Datamap {
    @Getter
    private final HashMap<String, Integer> inm = new HashMap<>();
    public Datamap put(String k, int v) {
        inm.put(k ,v);
        return this;
    }
}
