package br.ufrn.application;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.RAkEL;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.classifiers.trees.J48;

public class Main {

	public static void main(String[] args) throws Exception {
		String path = System.getProperty("user.dir");

		for (String metabase : bases()) {
			// String metabase = "combinador_ensemble_multirrotulo";

			System.out.println(metabase);
			
			String arff = path + "/arff/" + metabase + ".arff";
			String xml = path + "/xml/" + metabase + ".xml";

			MultiLabelInstances dataset = new MultiLabelInstances(arff, xml);

			RAkEL learner1 = new RAkEL(new LabelPowerset(new J48()));
			MLkNN learner2 = new MLkNN();

			Evaluator eval = new Evaluator();
			MultipleEvaluation results;

			results = eval.crossValidate(learner1, dataset, 10);
			saveResults(results.toString(), metabase + "_RAkEL");

			results = eval.crossValidate(learner2, dataset, 10);
			saveResults(results.toString(), metabase + "_MLkNN");
		}
	}

	public static void saveResults(String text, String fileName) {
		String path = System.getProperty("user.dir");

		try {
			File file = new File(path + "/output/" + fileName);

			FileWriter fileWriter = new FileWriter(file);
			fileWriter.write(text);
			fileWriter.close();
		} catch (IOException e) {
			System.out.println(e.toString());
		}
	}

	public static List<String> bases() {
		List<String> l = new ArrayList<String>();
		// N l.add("bibtex");
		l.add("birds");
		// N l.add("bookmarks");
		l.add("cal500");
		// l.add("corel5k");
		// N l.add("delicious");
		l.add("emotions");
		// N l.add("enron");
		l.add("flags");
		l.add("genbase");
		l.add("medical");
		return l;
	}
}