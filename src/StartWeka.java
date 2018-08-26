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

        Instances train = new Instances (breader); //breader (buffered reader) object is parsed into train object
        train.setClassIndex(train.numAttributes() -1); //Calling setClassIndex method

        breader.close(); //Good practice to always close your Buffered Reader when 'finished'

        NaiveBayes nB = new NaiveBayes(); //Creating Naive Bayes object (with a normal constructor)
        nB.buildClassifier(train); //Calling buildClassifier method and passing in train object - 'sent train file to nB by calling method buildClassifier
        Evaluation eval = new Evaluation(train); //Evaluation is a class - object 'eval' of Class Evaluation
        eval.crossValidateModel(nB, train, 10, new Random(1));
        //Calling .crossValidateModel(NaiveBayes classifier, train object, number of folds, random seed with int value 1)
        System.out.println(eval.toSummaryString("\nResults\n=======\n", true));
        System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
    }
}
