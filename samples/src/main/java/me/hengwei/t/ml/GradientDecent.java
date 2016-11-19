package me.hengwei.t.ml;

import java.util.ArrayList;
import java.util.List;

/**
 * demonstration for gradient decent for linear regression.
 * y = 2*x + 1; or f(w, b) = w*x + b;
 * with cost function
 * J(w, b) = Sigma(f(w, b) - y_label)^2
 * And iterator loop:
 * w* = w - alpha*(J(w,b))/
 * sample points:
 * ({0.5, 2})({0, 1})({1, 3})({-1, -1})({-2, -3})
 */
public class GradientDecent {

    static class PointData {
        double x;
        double y;
        public PointData(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }

    private static double learningRate = 0.003;
    private static double converageRate = 0.00000001;

    /**
     * training weight and bias;
     * ((wx + b) - y) * x --> weight derived.
     * ((wx + b) - y) * 1 --> bias derived.
     * @param trainSet
     */
    public static void trainLRWithGD(List<PointData> trainSet){
        if(trainSet == null) {
            return;
        }
        double w = 0, b = 0;
        double wConverge = 1, bConverge = 1;
        double sumWeightError = 0, sumBiasError = 0;
        while(wConverge> converageRate && bConverge > converageRate) {
            for(int i=0; i<trainSet.size(); i++) {
                PointData sample = trainSet.get(i);
                sumWeightError += (w * sample.x + b - sample.y) * sample.x;
                sumBiasError += (w * sample.x + b - sample.y);
            }
            double wUpdated, bUpdated;
            wUpdated = w - learningRate * sumWeightError;
            bUpdated = b - learningRate * sumBiasError;
            /* calculate converage rate */
            wConverge = (wUpdated - w) / wUpdated;
            bConverge = (bUpdated - b) / bUpdated;
            /* update w and b */
            w = wUpdated;
            b = bUpdated;
        }
        System.out.println("trained weight: " + w + "\t bias: " + b );
    }

    public static void main(String[] args) {
        List<PointData> trainSet = new ArrayList<PointData>();
        trainSet.add(new PointData(0.5, 2));
        trainSet.add(new PointData(0, 1));
        trainSet.add(new PointData(1, 3));
        trainSet.add(new PointData(-1, -1));
        trainSet.add(new PointData(-2, -3));
        GradientDecent.trainLRWithGD(trainSet);
    }

}
