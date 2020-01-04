package com.baeldung.deeplearning4j;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.local.transforms.AnalyzeLocal;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class Bank {
    public static void main(String[] args) throws Exception {
        final CSVRecordReader recordReader = createRecordReader();
        final Schema schema = createSchema();

        final DataAnalysis analysis = AnalyzeLocal.analyze(schema, recordReader);
        HtmlAnalysis.createHtmlAnalysisFile(analysis, new File("analysis.html"));
    }

    private static CSVRecordReader createRecordReader() throws IOException, InterruptedException {
        Random random = new Random();
        random.setSeed(0xC0FFEE);
        FileSplit inputSplit = new FileSplit(new ClassPathResource("churn/").getFile(), random);

        CSVRecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(inputSplit);
        return recordReader;
    }

    private static Schema createSchema() {
        return new Schema.Builder()
                .addColumnsInteger("Row Number", "Customer Id")
                .addColumnString("Surname")
                .addColumnInteger("Credit Score")
                .addColumnCategorical("Geography", "France", "Germany", "Spain")
                .addColumnCategorical("Gender", "Female", "Male")
                .addColumnsInteger("Age", "Tenure")
                .addColumnDouble("Balance")
                .addColumnInteger("Num Of Products")
                .addColumnCategorical("Has Credit Card", "0", "1")
                .addColumnCategorical("Is Active Member", "0", "1")
                .addColumnDouble("Estimated Salary")
                .addColumnCategorical("Exited", "0", "1")
                .build();
    }

}
