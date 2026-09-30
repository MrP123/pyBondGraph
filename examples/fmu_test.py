from fmpy import dump, simulate_fmu
from fmpy.validation import validate_fmu

import os
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

import numpy as np
signals = np.array([(0.0, 5.0), (1.0, 5.0)],
                   dtype=[('time', 'f8'), ('U_in', 'f8')])

# Static checks: modelDescription.xml schema, variable refs, causality/variability rules
problems = validate_fmu("BondGraphSlave.fmu")
print(f"Problems: {problems}")          # empty list == passed static validation

dump("BondGraphSlave.fmu")            # prints the interface Simulink will see

# Dynamic check: actually loads the binary and steps the solver
result = simulate_fmu("BondGraphSlave.fmu", stop_time=0.1, input=signals)
for r in result:
    print(r)