import java.io.BufferedReader; //Read arff file
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class StartWeka
{
    public static void main(String[] args) throws Exception
    {
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("/Users/alistairgj/Documents/GitHub/Weka/iris.arff")); //Reading the arff file

        Instances train = new Instances (breader);
        train.setClassIndex(train.numAttributes() -1);

        breader.close();

        NaiveBayes nB = new NaiveBayes();
        nB.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(nB, train, 10, new Random(1));
        System.out.println(eval.toSummaryString("\nResults\n=======\n", true));
        System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
    }
}
