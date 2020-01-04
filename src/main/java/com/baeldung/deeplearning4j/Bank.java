package com.baeldung.deeplearning4j;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.local.transforms.AnalyzeLocal;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.IOException;
import java.util.Random;

public class Bank {
    public static void main(String[] args) throws Exception {
        final CSVRecordReader recordReader = new CSVRecordReader();
        final FileSplit inputSplit = createInputSplit();
        recordReader.initialize(inputSplit);
        
        final Schema schema = createSchema();

        final DataAnalysis analysis = AnalyzeLocal.analyze(schema, recordReader);
        //HtmlAnalysis.createHtmlAnalysisFile(analysis, new File("analysis.html"));

        final TransformProcess transformProcess = getTransformProcess(schema, analysis);

        train(inputSplit, transformProcess);

    }

    private static FileSplit createInputSplit() throws IOException {
        Random random = new Random();
        random.setSeed(0xC0FFEE);
        return new FileSplit(new ClassPathResource("churn/").getFile(), random);
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

    private static TransformProcess getTransformProcess(Schema schema, DataAnalysis analysis) {
        return new TransformProcess.Builder(schema)
                .removeColumns("Row Number", "Customer Id", "Surname")
                .categoricalToOneHot("Geography", "Gender", "Has Credit Card", "Is Active Member")
                .integerToOneHot("Num Of Products", 1, 4)
                .normalize("Tenure", Normalize.MinMax, analysis)
                .normalize("Age", Normalize.Standardize, analysis)
                .normalize("Credit Score", Normalize.Log2Mean, analysis)
                .normalize("Balance", Normalize.Log2MeanExcludingMin, analysis)
                .normalize("Estimated Salary", Normalize.Log2MeanExcludingMin, analysis)
                .build();
    }

    private static void train(FileSplit inputSplit, TransformProcess transformProcess) throws IOException, InterruptedException {
        final RecordReaderDataSetIterator trainIterator = createTrainIterator(inputSplit, transformProcess);
        final MultiLayerConfiguration config = createNeuralNetConfig(transformProcess.getFinalSchema());

        final MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.fit(trainIterator, 59);
    }

    private static RecordReaderDataSetIterator createTrainIterator(FileSplit inputSplit, TransformProcess transformProcess) throws IOException, InterruptedException {
        TransformProcessRecordReader trainRecordReader = new TransformProcessRecordReader(new CSVRecordReader(), transformProcess);
        trainRecordReader.initialize(inputSplit);

        return new RecordReaderDataSetIterator.Builder(trainRecordReader, 80)
                .classification(transformProcess.getFinalSchema().getIndexOfColumn("Exited"), 2)
                .build();
    }

    private static MultiLayerConfiguration createNeuralNetConfig(Schema finalSchema) {
        return new NeuralNetConfiguration.Builder()
                .seed(0xC0FFEE)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .updater(new Adam.Builder().learningRate(0.001).build())
                .l2(0.0000316)
                .list(
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new OutputLayer.Builder(new LossMCXENT()).nOut(2).activation(Activation.SOFTMAX).build()
                )
                .setInputType(InputType.feedForward(finalSchema.numColumns() - 1))
                .build();
    }

}
