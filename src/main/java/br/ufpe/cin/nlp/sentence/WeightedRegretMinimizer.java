package br.ufpe.cin.nlp.sentence;

import ilog.concert.IloException;
import ilog.concert.IloIntVar;
import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

public class WeightedRegretMinimizer {
	private static IloCplex cplex;
	
	public static final boolean DEBUG = false;
	public static final int TIMELIMIT = 60;
	public static final int BIG_CONSTANT = Integer.MAX_VALUE/1000;
	
	static {
		try {
			cplex = new IloCplex();
			if (!DEBUG) {
				cplex.setOut(null);
			}
			cplex.setParam(IloCplex.DoubleParam.TiLim, TIMELIMIT);
		} catch (IloException ex) {
			ex.printStackTrace();
			throw new IllegalStateException("ERROR INITIALIZING CPLEX: " + ex.getMessage());
		}
	}
	
	public static double minWeightedRegretFor(double[] regrets, double[] outputWeigths) {
		try {
			IloNumVar[] pVars = new IloNumVar[regrets.length];
			for (int i = 0; i < pVars.length; i++) {
				pVars[i] = cplex.numVar(0.0, BIG_CONSTANT);
			}
			IloNumVar[] weights = new IloNumVar[regrets.length];
			for (int i = 0; i < weights.length; i++) {
				weights[i] = cplex.numVar(0.0, 1.0);
			}
			for (int k = 0; k < pVars.length; k++) {
				int j = regrets.length - 1;
				int i = j - 1;
				IloNumVar lmax = cplex.numVar(0, BIG_CONSTANT);
	
				final IloNumExpr prod1 = cplex.prod(regrets[j] , weights[j]);
				final IloNumExpr prod2 = cplex.prod(regrets[i], weights[i]);
				cplex.addGe(lmax, prod1);
				cplex.addGe(lmax, prod2);
				
				IloIntVar c = cplex.boolVar();
				cplex.addLe(cplex.diff(lmax, prod1), cplex.prod(c, BIG_CONSTANT));
				cplex.addLe(cplex.diff(lmax, prod2), cplex.prod(cplex.diff(1, c), BIG_CONSTANT));
	
				i--;
				while (i >= 0) {
					IloNumVar newmax = cplex.numVar(0, BIG_CONSTANT);
					cplex.addGe(newmax, lmax);
					final IloNumExpr prod = cplex.prod(regrets[i], weights[i]);
					cplex.addGe(newmax, prod);
					IloIntVar d = cplex.boolVar();
					cplex.addLe(cplex.diff(newmax, lmax), cplex.prod(d, BIG_CONSTANT));
					cplex.addLe(cplex.diff(newmax, prod), cplex.prod(cplex.diff(1, d), BIG_CONSTANT));
	
					lmax = newmax;
					i--;
				}
				cplex.addEq(pVars[k], lmax);
			}
			
			cplex.addEq(cplex.sum(weights), 1.0);
			IloNumExpr expressionObjective = cplex.sum(pVars);
			cplex.addMinimize(expressionObjective);
			double ret = -1.0;
			if (cplex.solve()) {
				if (cplex.getStatus() == IloCplex.Status.Optimal) {
					ret = cplex.getObjValue();
					if (outputWeigths != null) {
						double[] newWeights = cplex.getValues(weights);
						System.arraycopy(newWeights, 0, outputWeigths, 0, outputWeigths.length);
					}
				}
				
			}
			return ret;
		} catch (IloException ex) {
			throw new IllegalStateException("Error when trying to use optimizer", ex);
		} finally {
			try {
				cplex.clearModel();
			} catch (IloException e) {
				System.err.println("Error when trying to clear cplex model");
				e.printStackTrace();
			}
		}
	}
	
}
