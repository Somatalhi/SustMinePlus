param NBLOCKS;
param NPERIODS;
param NRESOURCE_SIDE_CONSTRAINTS;
param DISCOUNT_RATE;
set setBL = 0..NBLOCKS-1;
set setT = 0..NPERIODS-1;
set setSIDE = 0..NRESOURCE_SIDE_CONSTRAINTS-1;

param RESOURCE_CONSTRAINT_UB_LIMITS {setSIDE,setT};
param RESOURCE_CONSTRAINT_LB_LIMITS {setSIDE,setT};
param RESOURCE_CONSTRAINT_COEFFICIENTS {setBL, setSIDE} default 0;
param OBJECTIVE_FUNCTION {setBL};
set PREC{setBL};

var X{setBL,setT} binary;

maximize Profit: sum{t in setT} sum{b in setBL} (1.0/(1.0 + DISCOUNT_RATE))^t * OBJECTIVE_FUNCTION[b] * X[b,t];

subject to Resource { r in setSIDE, t in setT}:
		RESOURCE_CONSTRAINT_LB_LIMITS[r,t] <= sum{b in setBL} RESOURCE_CONSTRAINT_COEFFICIENTS[b,r] * X[b,t] <= RESOURCE_CONSTRAINT_UB_LIMITS[r,t];

subject to CliqueBlock {b in setBL}:
		sum{t in setT} X[b,t] <= 1;

subject to Precedence {b in setBL, i in PREC[b], t in setT}:
    sum{s in 0..t} X[b,s] <= sum{s in 0..t} X[i,s];



		
