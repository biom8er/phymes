# Example: Biochemical network simulator

A simple network describing the interconnection of molecules is simulated until a steady state is reached. The time axis is uniformly discretized and walked along until there is no further change in molecule amounts. If the change in any molecule amounts is greater than a given threshold, the step is repeated with a finer grained time delta. If the change in all molecules is below a given threshold, the time delta is increased on the next step.

In progress...