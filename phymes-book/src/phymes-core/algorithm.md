# Phymes algorithm

## Hypergraphs are powerful representations for the complexities of the real world

Hypergraphs are a generalization of graphs that can better represent the complexities of the real world. For example, biochemical networks can be represented as a directed hypergraph where nodes are molecules and hyperedges are reactions that catalyze the conversion of molecules into other molecules. The directionalty of the reaction is lost when represented as an indirected graph and the grouping of molecules involved in the reaction is lost when using directed graphs.

Mathematically, the biochemical reaction hypergraph can be represented as an incidence matrix where the rows corresponding to molecules, the columns correspond to reactions, and the entries are positive or negative integers corresponding the the number of molecules consumed or produced by the reaction.

While simplistic, this representation is already useful for decision science whereby one can optimize the molecules flow through the network using (mixed integer)-(non)-linear programming, for simulation whereby one can reverse parameter fit the network to data and then forward simulate the network using numerical integration solvers, for exploratory data science whereby unsupervised methods such as PCA, KNN clustering, etc can be applied to learn about the network topological and dynamic properties, and for forecasting whereby once can fit a hypergraph neural network to make predictions about future node, hyperedge, or hypergraph properties.

Hypergraphs can be further decomposed into binary and unary hyper edges. This decomposition better aligns with the binary and unary operators of computational graphs. This decomposition also aligns well with biochemical computational graphs where reactions can be broken down into steps describing the binding of individual molecules to a catalyst, the rearrangement of atoms, and the unbinding of individual molecules. When the exact steps are not known, a set of possible hypergraph configurations can be constructed and then weighted by their probability or used as the basis for follow up experiments to determine the exact steps.

## Phymes is a parallel and streaming hypergraph algorithm primitive

Phymes is designed around the concept of a hypergraph where subjects are nodes and tasks are hyperedges. Phymes uses a publish-subscribe message passing scheme to implement the hypergraph. Specifically, tasks subscribe to and publish on subjects. The addition of processor metrics, tracing, and the subject tables themselves provides transparency on all hypergraph computation steps.

The execution of tasks is done in parallel and Asynchronously. Tasks can either always subscribe to a subject or conditionally subscribe when a subject is updated. Tasks run only when all of their subscriptions are ready. Conditional subscription on updated subjects enables the coding of first order logic to control when tasks execute. Tasks then publish the results of their computation to subjects. The execution of tasks is coordinated through supersteps where ready tasks are ran in parallel.

Phymes incorporates decomposition of hyperedges. Tasks are composed of processes that iteratively operate over streams of subject messages. The chaining of processors enables the representation of complex computational graphs. The streaming of messages enables the scaling to massive data sets that do not fit into memory on a single node. The computation over streaming messages enables parallel and Asynchronous execution of tasks at each superstep.

Phymes provides the algorithmic basis and computational framework for implementing efficient and scalable algorithms for decision science, simulation, data science, and machine learning. In particular, phymes provides a cross platform, secure, performant, and transparent Agentic AI framework with support for complex ETL and data science function calls.