from odbAccess import *
from abaqusConstants import *
import numpy as np

"""
Script description:
Get the deformation gradient of the deformed configuration from abaqus.
"""
deformation_gradients = []

odbName = 'abaqus/circ_multistep_s100.odb'
myOdb = openOdb(odbName)
for stepName in myOdb.steps.keys():
    # step = myOdb.steps[stepName]
    frame = myOdb.steps[stepName].frames[-1]
    F_values = frame.fieldOutputs['EP'].values
    deformation_gradients.append(F_values)


deformation_gradients_array = np.array(deformation_gradients)
np.save('abaqus/deformation_gradients.npy', deformation_gradients_array)