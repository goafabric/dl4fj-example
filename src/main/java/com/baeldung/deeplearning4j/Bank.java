package com.baeldung.deeplearning4j;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.Random;

public class Bank {
    public static void main(String[] args) throws IOException, InterruptedException {
        Random random = new Random();
        random.setSeed(0xC0FFEE);
        FileSplit inputSplit = new FileSplit(new ClassPathResource("Churn_Modelling.csv").getFile(), random);

        CSVRecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(inputSplit);
    }
}
