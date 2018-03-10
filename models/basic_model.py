
# coding: utf-8

# # Mexican Flat Slab model
# 
# * Mor-Sz dist at 40 Ma, ~2000 km
# * Mor-Sz dist at 0 Ma, ~600 km
# * Mor average Vel ~ 2 cm/y
# * UP vel ~ 2 cm/y
# * Sp vel (35 - 10 Ma) ~ 8 cm/y
# * Sp vel (10 - 0 Ma) ~ 3 cm/y

# In[1]:


#!apt-cache policy petsc-dev


# In[2]:


#If run through Docker we'll point at the local 'unsupported dir.'
#On hpc, the path should also include a directory holding the unsupported_dan.
import sys

#this does't actually need to be protected. More a reminder it's an interim measure
try:
    sys.path.append('../../unsupported')
    sys.path.append('../../UWsubduction/')
    #use this block to point to a local version of UWsubduction

except:
    pass


# In[3]:


import os

import numpy as np
import underworld as uw
from underworld import function as fn
import glucifer
from easydict import EasyDict as edict
import networkx as nx
import operator

import warnings; warnings.simplefilter('ignore')


# In[4]:


import UWsubduction as usub
import UWsubduction.params as params 
import UWsubduction.utils as utils


# In[5]:


#load in parent stuff
#%load_ext autoreload
#import nb_load_stuff
#from tectModelClass import *
#from unsupported_dan.UWsubduction.model import *


# In[6]:


#TectModel


# In[7]:


from unsupported_dan.utilities.interpolation import nn_evaluation


# ## Create output dir structure

# In[8]:


############
#Model letter and number
############


#Model letter identifier demarker
Model = "T"

#Model number identifier demarker:
ModNum = 0

#Any isolated letter / integer command line args are interpreted as Model/ModelNum

if len(sys.argv) == 1:
    ModNum = ModNum 
elif sys.argv[1] == '-f': #
    ModNum = ModNum 
else:
    for farg in sys.argv[1:]:
        if not '=' in farg: #then Assume it's a not a paramter argument
            try:
                ModNum = int(farg) #try to convert everingthing to a float, else remains string
            except ValueError:
                Model  = farg
                
                
###########
#Standard output directory setup
###########

outputPath = "results" + "/" +  str(Model) + "/" + str(ModNum) + "/" 
valuesPath = outputPath + 'values/'
filePath = outputPath + 'files/'
#checkpointPath = outputPath + 'checkpoint/'
dbPath = outputPath + 'gldbs/'
xdmfPath = outputPath + 'xdmf/'
outputFile = 'results_model' + Model + '_' + str(ModNum) + '.dat'

if uw.rank()==0:
    # make directories if they don't exist
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    if not os.path.isdir(valuesPath):
        os.makedirs(valuesPath)
    if not os.path.isdir(dbPath):
        os.makedirs(dbPath)
    if not os.path.isdir(filePath):
        os.makedirs(filePath)
    if not os.path.isdir(xdmfPath):
        os.makedirs(xdmfPath)
        
uw.barrier() #Barrier here so no procs run the check in the next cell too early


# ## Parameters / Scaling
# 
# * For more information see, `UWsubduction/Background/scaling`
# 

# In[9]:


#from unsupported_dan.UWsubduction.minimal_example import UnitRegistry
u = params.UnitRegistry


# In[10]:


#pd refers to dimensional paramters
pd = edict({})

#Main physical paramters (thermal convection parameters)
pd.refDensity = 3300.* u.kilogram / u.meter**3                    #reference density
pd.refGravity = 9.8* u.metre / u.second**2                        #surface gravity
pd.refDiffusivity = 1e-6 *u.metre**2 / u.second                   #thermal diffusivity
pd.refExpansivity = 3e-5/u.kelvin                                 #surface thermal expansivity
pd.refViscosity = 1e20* u.pascal* u.second
pd.refLength = 2900*u.km
pd.gasConstant = 8.314*u.joule/(u.mol*u.kelvin)                   #gas constant
pd.specificHeat = 1250.4*u.joule/(u.kilogram* u.kelvin)           #Specific heat (Jkg-1K-1)
pd.potentialTemp = 1573.*u.kelvin                                 #mantle potential temp (K)
pd.surfaceTemp = 273.*u.kelvin                                    #surface temp (K)
#these are the shifted temps, which will range from 0 - 1 in the dimensionless system
pd.potentialTemp_ = pd.potentialTemp - pd.surfaceTemp
pd.surfaceTemp_ = pd.surfaceTemp - pd.surfaceTemp
#main rheology parameters
pd.cohesionMantle = 20.*u.megapascal                              #mantle cohesion in Byerlee law
pd.frictionMantle = u.Quantity(0.1)                                           #mantle friction coefficient in Byerlee law (tan(phi))
pd.frictionMantleDepth = pd.frictionMantle*pd.refDensity*pd.refGravity
pd.diffusionPreExp = 5.34e-10/u.pascal/u.second                   #pre-exp factor for diffusion creep
pd.diffusionEnergy = 3e5*u.joule/(u.mol)
pd.diffusionEnergyDepth = 3e5*u.joule/(u.mol*pd.gasConstant)
pd.diffusionVolume=5e-6*u.meter**3/(u.mol)
pd.diffusionVolumeDepth=5e-6*pd.refDensity.magnitude*pd.refGravity.magnitude*u.joule/(u.mol*pd.gasConstant*u.meter)
pd.viscosityFault = 2e19*u.pascal  * u.second
pd.adiabaticTempGrad = (pd.refExpansivity*pd.refGravity*pd.potentialTemp)/pd.specificHeat
pd.yieldStressMax=200*u.megapascal
pd.lowerMantleViscFac = u.Quantity(20.0)



# In[11]:


md = edict({})
#Model geometry, and misc Lengths used to control behaviour
md.depth=1000*u.km                                                #Model Depth
md.aspectRatio=5.0
#lengths, factors relating to subduction fault behaviour
md.faultViscDepthTaperStart = 100*u.km
md.faultViscDepthTaperWidth = 30*u.km
md.faultViscHorizTaperStart = 150*u.km
md.faultViscHorizTaperWidth = 150*u.km
md.faultThickness = 10.*u.km
md.faultLocFac = 1.                                                #this is the relative location of the fault in terms of the fault thickess from the top of slab
md.faultDestroyDepth = 300*u.km
md.lowerMantleDepth=660.*u.km
md.lowerMantleTransWidth=100.*u.km
#Slab and plate init. parameters
md.subZoneLoc=-100*u.km                                           #X position of subduction zone...km
md.slabInitMaxDepth=150*u.km
md.radiusOfCurv = 200.*u.km                                        #radius of curvature
md.slabAge=20.*u.megayears                                      #age of subduction plate at trench
md.opAgeAtTrench=15.*u.megayears                                        #age of op
#numerical and computation params
md.res=24
md.ppc=25                                                         #particles per cell
md.elementType="Q1/dQ0"
md.refineHoriz = True
md.refineVert = True
md.meshRefineFactor = 0.7
md.nltol = 0.01
md.druckerAlpha = 1.
md.penaltyMethod = True
md.buoyancyFac = 1.0
md.viscosityMin = 1e18* u.pascal * u.second
md.viscosityMax = 1e24* u.pascal * u.second
#wedge stuff
md.wedgeViscosity = 2e20* u.pascal * u.second
md.wedgeShallow=45*u.km
md.wedgeDeep=150*u.km
md.wedgeThickness = 200*u.km
md.turnOnWedge = 20*u.megayears
md.turnOffVels = False



# In[12]:


#first check for commandLineArgs:

sysArgs = sys.argv
utils.easy_args(sysArgs, pd)
utils.easy_args(sysArgs, md)


#for some reason I'm changing the names...
modelDict_dim = md
paramDict_dim = pd

#print(md.depth, md.turnOffVels, type(md.turnOffVels),pd.viscosityFault


# In[13]:




#define some more concise names
sca = params.sub_scaling
ndimlz = sca.nonDimensionalize
#build the dimensionless paramter / model dictionaries
ndp = params.build_nondim_dict(paramDict_dim  , sca)   
md = params.build_nondim_dict(modelDict_dim  , sca)
ur = u

assert ndimlz(paramDict_dim.refLength) == 1.0

# changes to base params: (These will keep changing if the notebook is run again without restarting!)
#md.faultThickness *= 1.5 #15 km
#md.res = 48


#print(md.depth, md.turnOffVels, type(md.turnOffVels),pd.viscosityFault)
print( type(md.turnOffVels))


# In[14]:


#1./ndimlz(1.*ur.megapascal)
1./ndimlz(1.*ur.megayear)


# In[15]:


#delt = 2000*ur.kilometer/(7*ur.centimeter/ur.year)
#delt.to(ur.megayear)


# ## Build / refine mesh, Stokes Variables

# In[16]:


#(ndp.rightLim - ndp.leftLim)/ndp.depth
#md.res = 64


# In[17]:


yres = int(md.res)
xres = int(md.res*6) 

halfWidth = 0.5*md.depth*md.aspectRatio 

minCoord_    = (-1.*halfWidth, 1. - md.depth) 
maxCoord_    = (halfWidth, 1.)

mesh = uw.mesh.FeMesh_Cartesian( elementType = (md.elementType),
                                 elementRes  = (xres, yres), 
                                 minCoord    = minCoord_, 
                                 maxCoord    = maxCoord_) 

velocityField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2)
pressureField   = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 ) 
    

velocityField.data[:] = 0.
pressureField.data[:] = 0.
temperatureField.data[:] = 0.
temperatureDotField.data[:] = 0.


# In[18]:


#mesh.reset() #call to reset mesh nodes to original locations

if md.refineHoriz:
    
    with mesh.deform_mesh():
        
        normXs = 2.*mesh.data[:,0]/(mesh.maxCoord[0] - mesh.minCoord[0])
        mesh.data[:,0] = mesh.data[:,0] * np.exp(md.meshRefineFactor*normXs**2) / np.exp(md.meshRefineFactor*1.0**2)    
    
if md.refineVert:

    with mesh.deform_mesh():
        
        mesh.data[:,1] = mesh.data[:,1] - 1.0

        normYs = -1.*mesh.data[:,1]/(mesh.maxCoord[1] - mesh.minCoord[1])
        mesh.data[:,1] = mesh.data[:,1] * np.exp(md.meshRefineFactor*normYs**2)/np.exp(md.meshRefineFactor*1.0**2)

        mesh.data[:,1] = mesh.data[:,1] + 1.0


# In[19]:


mesh.minCoord[0], mesh.maxCoord[0]


# In[20]:


test = 1000*ur.kilometer/ (5*ur.cm/ur.year)
test.to(ur.megayear)


# ## Build plate model

# In[21]:


endTime = ndimlz(35*ur.megayear) 
refVel = ndimlz(2*ur.cm/ur.year)
plateModelDt = ndimlz(0.1*ur.megayear)


# In[22]:


#velocities of the plates (1 - 3) ams well as the plate boundary (1,2)
vp1= ndimlz(0.*ur.centimeter/ur.year )

vp3start= ndimlz(-3.*ur.centimeter/ur.year )
vp3end= ndimlz(-1.0*ur.centimeter/ur.year )

vb12= ndimlz(1.5*ur.centimeter/ur.year )


vp2start= ndimlz(8.*ur.centimeter/ur.year )
vp2end= ndimlz(2.*ur.centimeter/ur.year )



# In[23]:


tm = usub.TectonicModel(mesh, 0, endTime, plateModelDt)

velsP2 = np.linspace(vp2start, vp2end, len(tm.times))
velsP3 = np.linspace(vp3start, vp3end, len(tm.times))



tm.add_plate(1, velocities=vp1)
#tm.add_plate(2, velocities=vp2start)
tm.add_plate(2, velocities=velsP2)
tm.add_plate(3, velocities=velsP3)


if md.turnOffVels:
    ix_ = np.argmin(np.abs(tm.times - md.turnOnWedge))
    tm.node[2]['velocities'][ix_:] = np.nan


# In[24]:


#tm.node[2]['velocities']


# In[25]:


#switch off SP velocity
#velsFalse_ = np.zeros(len(tm.times))
#velsFalse_.fill(np.nan)
#tm.node[2]['velocities'] = velsFalse_


# In[26]:


ridgeLoc = -0.7
subLoc = ridgeLoc  + ndimlz(2100.*ur.kilometer)

tm.add_left_boundary(1, plateInitAge=md.slabAge/3., velocities=False)
tm.add_ridge(1,2, ridgeLoc, velocities=vb12)
tm.add_subzone(2, 3, subLoc, subInitAge=md.slabAge, upperInitAge=md.opAgeAtTrench)
tm.add_right_boundary(3, plateInitAge=0.0, velocities=False)


# In[27]:


#((1400*ur.kilometer)/(35*ur.megayear)).to(ur.cm/ ur.year )


# ## Build plate age / temperature Fns

# In[28]:


pIdFn = tm.plate_id_fn()
pAgeDict = tm.plate_age_fn() 

fnAge_map = fn.branching.map(fn_key = pIdFn , 
                          mapping = pAgeDict )

#fig = glucifer.Figure(figsize=(600, 300))
#fig.append( glucifer.objects.Surface(tm.mesh, fnAge_map ))
#fig.show()


# In[29]:


#ndp.potentialTemp


# In[30]:


coordinate = fn.input()
depthFn = mesh.maxCoord[1] - coordinate[1]

platethickness = 2.32*fn.math.sqrt(1.*fnAge_map )  

halfSpaceTemp = ndp.potentialTemp_*fn.math.erf((depthFn)/(2.*fn.math.sqrt(1.*fnAge_map)))

plateTempProxFn = fn.branching.conditional( ((depthFn > platethickness, ndp.potentialTemp_ ), 
                                           (True,                      halfSpaceTemp)  ))



# In[31]:


#fig = glucifer.Figure(figsize=(600, 300))
#fig.append( glucifer.objects.Surface(tm.mesh, plateTempProxFn, onMesh = True))
#fig.show()


# ## Make swarm and Swarm Vars

# In[32]:


swarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
layout = uw.swarm.layouts.PerCellRandomLayout(swarm=swarm, particlesPerCell=int(md.ppc))
swarm.populate_using_layout( layout=layout ) # Now use it to populate.
proxyTempVariable = swarm.add_variable( dataType="double", count=1 )
proximityVariable      = swarm.add_variable( dataType="int", count=1 )
signedDistanceVariable = swarm.add_variable( dataType="double", count=1 )
wedgeVariable = swarm.add_variable( dataType="int", count=1 )

#
proxyTempVariable.data[:] = 1.0
proximityVariable.data[:] = 0
signedDistanceVariable.data[:] = 0.0
wedgeVariable.data[:] = 0


# ## Create tmUwMap

# In[33]:


#Now we have built are primary FEM / Swarm objects, we collect some of these in a dictionary,
#to provide a consistent form to pass to methods of TectModel

tmUwMap = usub.tm_uw_map([], velocityField, swarm, 
                    signedDistanceVariable, proxyTempVariable, proximityVariable)


# ## Make slab perturbation and subduction interface
# 
# * For more information see, `UWsubduction/Background/interface2D`

# In[34]:


def circGradientFn(S):
    if S == 0.:
        return 0.
    elif S < md.radiusOfCurv:
        return max(-S/np.sqrt((md.radiusOfCurv**2 - S**2)), -1e3)
    else:
        return -1e5
    
    
def circGradientFn2(S):
    if S == 0.:
        return 0.
    elif S < 1.*md.radiusOfCurv:
        return max(-S/np.sqrt((md.radiusOfCurv**2 - S**2)), -3.)
    else:
        return -3.

def circGradientFn3(S):
    if S < 1.*md.radiusOfCurv:
        return circGradientFn2(S)
    else:
        return min(circGradientFn2(2.6*md.radiusOfCurv - S), -0.1)
    


# In[35]:


#define fault particle spacing
ds = (tm.maxX - tm.minX)/(8.*tm.mesh.elementRes[0])

fCollection = usub.interface_collection([])

deepDepth = ndimlz(150.*ur.kilometer)

for e in tm.undirected.edges():
    if tm.is_subduction_boundary(e):
        usub.build_slab_distance(tm, e, circGradientFn3, deepDepth, tmUwMap)
        #build_slab_distance(tm, e, circGradientFn, md.slabInitMaxDepth, tmUwMap)        
        fb = usub.build_fault(tm, e, circGradientFn3, md.faultThickness , deepDepth, ds, md.faultThickness, tmUwMap)
        #fb = build_fault(tm, e, circGradientFn, md.faultThickness , md.slabInitMaxDepth, ds, md.faultThickness, tmUwMap)

        fCollection.append(fb)

#
usub.build_slab_temp(tmUwMap, ndp.potentialTemp_, md.slabAge)
fnJointTemp = fn.misc.min(proxyTempVariable,plateTempProxFn)

#And now reevaluate this guy on the swarm
proxyTempVariable.data[:] = fnJointTemp.evaluate(swarm)


# In[36]:


#fig = glucifer.Figure(figsize=(600, 300))
#fig.append( glucifer.objects.Points(swarm, proxyTempVariable))
#fig.append( glucifer.objects.Points(fb.swarm))
#fig.show()
#fig.save_database('test.gldb')


# ##  Define subduction fault evolution (rebuild/destroy)
# 
# 
# In this section we setup some functions to help manage the spatial distribution of faults

# In[37]:


# Setup a swarm to define the replacment positions

fThick= fCollection[0].thickness

faultloc = 1. - md.faultThickness*md.faultLocFac

allxs = np.arange(mesh.minCoord[0], mesh.maxCoord[0], ds )[:-1]
allys = (mesh.maxCoord[1] - fThick)*np.ones(allxs.shape)

faultMasterSwarm = uw.swarm.Swarm( mesh=mesh )
dummy =  faultMasterSwarm.add_particles_with_coordinates(np.column_stack((allxs, allys)))
del allxs
del allys



##What are we doing here??

#this will be used to actively remove any fault particles around the ridge
faultRmfn = tm.variable_boundary_mask_fn(distMax = ndimlz(100*ur.km),
                                           distMin = ndimlz(20*ur.km), 
                                           relativeWidth = 0.1,
                                           boundtypes='ridge')



#this don;t need to mask the fault addition at all, let the removal do the work
#faultAddFn =  faultRmfn 
faultAddFn =  fn.misc.constant(True)

###The following mask function provide a way of building velocity conditions within the plates,
#This removes the regions around the ridges from the velocity nodes
velMask1 = tm.t2f(tm.ridge_mask_fn(dist=ndimlz(25*ur.km)))
#This removes the regions around the subduction from the velocity nodes
velMask2= tm.t2f(tm.subduction_mask_fn(dist=md.faultViscHorizTaperStart))
#combine
velMaskFn = operator.and_( velMask1,  velMask2)


#order is very important here
#dummy = remove_fault_drift(fCollection, faultloc)
#dummy = pop_or_perish(tm, fCollection, faultMasterSwarm, faultAddFn , ds)
#dummy = remove_faults_from_boundaries(tm, fCollection, faultRmfn )


# In[38]:


dummy = usub.remove_fault_drift(fCollection, faultloc)
dummy = usub.pop_or_perish(tm, fCollection, faultMasterSwarm, faultAddFn , ds)
dummy = usub.remove_faults_from_boundaries(tm, fCollection, faultRmfn )


# In[39]:


#maskFn_ = tm.t2f(faultRmfn)
#pIdFn = tm.plate_id_fn(maskFn=maskFn_)


# In[40]:


#def buildPlateBoundMarkerSwarms(tectModel):
#    boundSwarmList = []
#    for e in tectModel.undirected.edges():
#        loc = tectModel.get_bound_loc(e)
#        #print(loc)
#        points = np.array([loc, tectModel.mesh.maxCoord[1] - 1e-6])
#        _swarm = uw.swarm.Swarm(tectModel.mesh)
#        _swarm.add_particles_with_coordinates(np.expand_dims(points, 0))
#        boundSwarmList.append(_swarm)
#    return boundSwarmList


# In[41]:


#boundSwarmList = buildPlateBoundMarkerSwarms(tm)


# In[42]:


#fig = glucifer.Figure(figsize=(600, 300))
#fig.append( glucifer.objects.Surface(tm.mesh, velMaskFn, onMesh=True))
#for s in boundSwarmList:
##    fig.append( glucifer.objects.Points(s, pointSize=5))
#fig.show()
#fig.save_database('test.gldb')


# ## Proximity
# 
# 

# In[43]:


proximityVariable.data[:] = 0


# In[44]:


for f in fCollection:
    f.rebuild()
    f.set_proximity_director(swarm, proximityVariable, searchFac = 2., locFac=1.0)


# In[45]:


#update_faults()


# In[46]:


#figProx = glucifer.Figure(figsize=(960,300) )
#figProx.append( glucifer.objects.Points(swarm , proximityVariable))
#figProx.append( glucifer.objects.Surface(mesh, velMaskFn))

#for f in fCollection:
#    figProx.append( glucifer.objects.Points(f.swarm, pointSize=5))
#figProx.show()


#figProx.save_database('test.gldb')


# In[47]:


#testMM = fn.view.min_max(uw.function.input(f.swarm.particleCoordinates))
#dummyFn = testMM.evaluate(tWalls)


# ## Define Wedge region

# In[48]:


wedgeVariable.data[:] = 0
def build_wedge_region():
    
    wedgeVariable.data[:] = 0

    sd, pts0 = fCollection[0].compute_signed_distance(swarm.particleCoordinates.data, 10.)

    #1./ndimlz(1*ur.megayear)
    wedgeVariable.data[np.logical_and(sd<(-1.*md.faultThickness),sd>-1.*md.wedgeThickness)] = 1
    mask = np.where(np.logical_or((1. - swarm.particleCoordinates.data[:,1]) < md.wedgeShallow, 
                                    (1. - swarm.particleCoordinates.data[:,1]) > md.wedgeDeep)) [0] 
    wedgeVariable.data[mask] = 0

    #And outside of the x lim
    #mask2 = fCollection[0].swarm.particleCoordinates.data[:,1] > (1. - md.wedgeDeep)
    #if mask2.sum():
    #    maxXWedge = np.max(fCollection[0].swarm.particleCoordinates.data[:,0][mask2]) + 2*ds
    #else:
    #    maxXWedge = mesh.maxCoord[0]
    
    #md.faultThickness, md.wedgeThickness
    
    #this relies on using the max fault depth as the max wedge depth.
    minmax_coordx = fn.view.min_max(fn.coord()[0])
    ignore = minmax_coordx.evaluate(fCollection[0].swarm)
    leftExt = minmax_coordx.min_global()
    rightExt = minmax_coordx.max_global()
  
    mask3 = swarm.particleCoordinates.data[:,0] > rightExt

    if mask3.sum():
        wedgeVariable.data[mask3 ] = 0


# In[49]:


#build this guy here so it has contains both 0s and 1s - 
#otherwise the mappign dictionary may fail

build_wedge_region()


# In[50]:


#figProx = glucifer.Figure(figsize=(960,300) )
#figProx.append( glucifer.objects.Points(swarm ,wedgeVariable))
#for f in fCollection:
#    figProx.append( glucifer.objects.Points(f.swarm))

#figProx.show()


# In[51]:


#figProx.save_database("test.gldb")


# ## Prescribed velocity

# In[52]:


def set_vel_return_nodes(time, maskFn):
    
    """
    globals:
    velocityField
    
    """
    
    nodes = tm.plate_vel_node_fn(time, maskFn = maskFn)
    
    #4 parallel safety
    if not nodes.shape[0]:
        return np.array([])
        
    pIdFn = tm.plate_id_fn()
    #velMapFn = tm.plateVelFn(testTime, pIdFn)
    velMapFn = tm.plateVelFn(time, pIdFn)
    locs = tm.mesh.data[nodes]

    #Now set the veolcity on the nodes
    velocityField.data[nodes, 0] = velMapFn.evaluate(locs)[:,0]
    return nodes
    


# In[53]:


vXnodes = set_vel_return_nodes(0., velMaskFn)


# In[54]:


#np.empty(0), 
#test = tm.mesh.specialSets['MaxJ_VertexSet']data.shape


# In[55]:


#check
#%pylab inline
#tWalls = tm.mesh.specialSets['MaxJ_VertexSet']


#fig, ax = plt.subplots(figsize=(10, 2))
#plt.plot(mesh.data[tWalls.data][:,0], velocityField.data[tWalls.data][:,0])#
#plt.scatter(mesh.data[vXnodes ][:,0], np.zeros(len(mesh.data[vXnodes ][:,0])), s = 0.5, c = 'k')
#ax.hlines(500, tm.minX, tm.maxX, linestyles='--')


# ## Project the swarm 'proxy temp' to mesh

# In[56]:


projectorMeshTemp= uw.utils.MeshVariable_Projection( temperatureField, proxyTempVariable , type=0 )
projectorMeshTemp.solve()


# ## Boundary conditions

# In[57]:


iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]

tWalls = mesh.specialSets["MaxJ_VertexSet"]
bWalls = mesh.specialSets["MinJ_VertexSet"]
rWalls = mesh.specialSets["MaxI_VertexSet"]

#velBC  = uw.conditions.DirichletCondition( variable        = velocityField, 
#                                           indexSetsPerDof = (iWalls, jWalls) )


# In[58]:


#vXnodes


# In[59]:


def build_velBcs(nodes):
    
    velnodeset = mesh.specialSets["Empty"]
    velnodeset += nodes

    
    velBC  = uw.conditions.DirichletCondition( variable        = velocityField, 
                                           indexSetsPerDof = (iWalls + velnodeset, jWalls) )
    
    return velBC


# In[60]:


velBC = build_velBcs(vXnodes)


# In[61]:


#Ridges enforced
dirichTempBC = uw.conditions.DirichletCondition(     variable=temperatureField, 
                                              indexSetsPerDof=(tWalls + rWalls,) )


###If we want thermal ridges fixed
temperatureField.data[rWalls.data] = ndp.potentialTemp_


# In[62]:


## Reassert the tempBCS

temperatureField.data[tWalls.data] = ndp.surfaceTemp_
temperatureField.data[bWalls.data] = ndp.potentialTemp_


# ## Bouyancy

# In[63]:


temperatureFn = temperatureField


# Now create a buoyancy force vector using the density and the vertical unit vector. 
thermalDensityFn = md.buoyancyFac*params.rayleighNumber*(1. - temperatureFn)

gravity = ( 0.0, -1.0 )

buoyancyMapFn = thermalDensityFn*gravity


# ## Rheology

# In[64]:


symStrainrate = fn.tensor.symmetric( 
                            velocityField.fn_gradient )

#Set up any functions required by the rheology
strainRate_2ndInvariant = fn.tensor.second_invariant( 
                            fn.tensor.symmetric( 
                            velocityField.fn_gradient ))



def safe_visc(func, viscmin=md.viscosityMin, viscmax=md.viscosityMax):
    return fn.misc.max(viscmin, fn.misc.min(viscmax, func))


# In[65]:


#tm.subZoneAbsDistFn()


# In[66]:


#Interface rheology extent

subZoneDistfn = tm.subZoneAbsDistFn(nonSpVal = 0.0, upper=False)
#subZoneDistfn = tm.subZoneAbsDistFn(upper=True)


faultHorizTaperFn  = usub.cosine_taper(subZoneDistfn, 
                                  md.faultViscHorizTaperStart, md.faultViscHorizTaperWidth)
faultDepthTaperFn = usub.cosine_taper(depthFn, 
                                 md.faultViscDepthTaperStart, md.faultViscDepthTaperWidth)


# In[67]:


#fig = glucifer.Figure(figsize=(960,300))

#fig.append( glucifer.objects.Points(swarm, faultRheologyFn,  logScale=True))
#fig.append( glucifer.objects.Surface(mesh,  faultHorizTaperFn))

#fig.show()
#fig.save_database('test.gldb')


# In[68]:


#temperatureField, 

#ndp.diffusionPreExp, ndp.diffusionVolumeDepth, ndp.diffusionEnergyDepth, ndp.surfaceTemp
#(1./ndp.diffusionPreExp)
#md.faultViscHorizTaperWidth*2900.

#5/4.


# In[69]:




adiabaticCorrectFn = depthFn*ndp.adiabaticTempGrad
dynamicPressureProxyDepthFn = pressureField/params.pressureDepthGrad
druckerDepthFn = fn.misc.max(0.0, depthFn + md.druckerAlpha*(dynamicPressureProxyDepthFn))

#Diffusion Creep
diffusionUM = (1./ndp.diffusionPreExp)*    fn.math.exp( ((ndp.diffusionEnergyDepth +                    (depthFn*ndp.diffusionVolumeDepth))/((temperatureFn+ adiabaticCorrectFn + ndp.surfaceTemp))))

diffusionUM =     safe_visc(diffusionUM)
    
diffusionLM = ndp.lowerMantleViscFac*(1./ndp.diffusionPreExp)*    fn.math.exp( ((ndp.diffusionEnergyDepth +                    (depthFn*ndp.diffusionVolumeDepth))/((temperatureFn+ adiabaticCorrectFn + ndp.surfaceTemp))))

#diffusionLM =     safe_visc(diffusionLM)


transitionZoneTaperFn = usub.cosine_taper(depthFn, md.lowerMantleDepth - 0.5*md.lowerMantleTransWidth , md.lowerMantleTransWidth )


mantleCreep = diffusionUM*(1. - transitionZoneTaperFn) + transitionZoneTaperFn*diffusionLM

#Define the mantle Plasticity
ys =  ndp.cohesionMantle + (druckerDepthFn*ndp.frictionMantleDepth)
ysf = fn.misc.min(ys, ndp.yieldStressMax)
yielding = ysf/(2.*(strainRate_2ndInvariant) + 1e-15) 

mantleRheologyFn =  safe_visc(mantleCreep*yielding/(mantleCreep + yielding), 
                              viscmin=md.viscosityMin, viscmax=md.viscosityMax)


faultViscosityFn = ndp.viscosityFault

faultRheologyFn =  faultViscosityFn*(1. - faultDepthTaperFn) +                         faultDepthTaperFn*mantleRheologyFn + faultHorizTaperFn*mantleRheologyFn


# In[70]:


#(md.lowerMantleDepth - 0.5*md.lowerMantleTransWidth )*2900.0

#params.pressureDepthGrad


# In[71]:


#Here's how we include the wedge 
mantleRheologyFn_ = fn.branching.map( fn_key = wedgeVariable,
                                 mapping = {0:mantleRheologyFn,
                                            1:mantleRheologyFn} )


# In[72]:


#viscconds = ((proximityVariable == 0, mantleRheologyFn),
#             (True, interfaceViscosityFn ))

#viscosityMapFn = fn.branching.conditional(viscconds)
#viscosityMapFn = mantleRheologyFn


viscosityMapFn = fn.branching.map( fn_key = proximityVariable,
                             mapping = {0:mantleRheologyFn_,
                                        2:faultRheologyFn} )


# In[73]:


#md.wedgeViscosity


# In[74]:


#fig.save_database('test.gldb')


# In[75]:


#ndimlz(1.0*ur.megayear)


# ## Stokes

# In[76]:


surfaceArea = uw.utils.Integral(fn=1.0,mesh=mesh, integrationType='surface', surfaceIndexSet=tWalls)
surfacePressureIntegral = uw.utils.Integral(fn=pressureField, mesh=mesh, integrationType='surface', surfaceIndexSet=tWalls)

NodePressure = uw.mesh.MeshVariable(mesh, nodeDofCount=1)
Cell2Nodes = uw.utils.MeshVariable_Projection(NodePressure, pressureField, type=0)
Nodes2Cell = uw.utils.MeshVariable_Projection(pressureField, NodePressure, type=0)

def smooth_pressure(mesh):
    # Smooths the pressure field.
    # Assuming that pressure lies on the submesh, do a cell -> nodes -> cell
    # projection.

    Cell2Nodes.solve()
    Nodes2Cell.solve()

# a callback function to calibrate the pressure - will pass to solver later
def pressure_calibrate():
    (area,) = surfaceArea.evaluate()
    (p0,) = surfacePressureIntegral.evaluate()
    offset = p0/area
    pressureField.data[:] -= offset
    smooth_pressure(mesh)


# In[77]:


stokes = uw.systems.Stokes( velocityField  = velocityField, 
                                   pressureField  = pressureField,
                                   conditions     = [velBC,],
                                   fn_viscosity   = viscosityMapFn, 
                                   fn_bodyforce   = buoyancyMapFn )


# In[78]:


solver = uw.systems.Solver(stokes)

solver.set_inner_method("mumps")
solver.options.scr.ksp_type="cg"
solver.set_penalty(1.0e7)
solver.options.scr.ksp_rtol = 1.0e-4


# In[79]:


solver.solve(nonLinearIterate=True, nonLinearTolerance=md.nltol, callback_post_solve = pressure_calibrate)
solver.print_stats()


# In[80]:


#velocityField.data.max()


# ## Advection - Diffusion

# In[79]:


advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureFn, 
                                         phiDotField    = temperatureDotField, 
                                         velocityField  = velocityField,
                                         fn_sourceTerm    = 0.0,
                                         fn_diffusivity = ndp.refDiffusivity, 
                                         #conditions     = [neumannTempBC, dirichTempBC] )
                                         conditions     = [ dirichTempBC] )


# ## Swarm Advector

# In[80]:


advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )


# In[81]:


population_control = uw.swarm.PopulationControl(swarm, deleteThreshold=0.006, 
                                                splitThreshold=0.25,maxDeletions=1, maxSplits=3, aggressive=True,
                                                aggressiveThreshold=0.9, particlesPerCell=int(md.ppc))


# ## Set up a midplane swarm

# In[180]:


#midplane.swarm.particleCoordinates.data


# In[83]:


spId = 2
midPlaneDepth = ndimlz(25.*ur.kilometer + 0.5*(modelDict_dim.faultThickness))

midPlaneXs = np.arange(tm.get_boundaries(2)[0] + 2.*ds, tm.get_boundaries(2)[1] - 2.*ds, ds)
midPlaneYs = np.ones(len(midPlaneXs)) * (1. - midPlaneDepth)
midplane = usub.interface2D(mesh, velocityField, midPlaneXs, midPlaneYs,1.0, 2)

mCollection = usub.interface_collection([])
mCollection.append(midplane)


allxs = np.arange(mesh.minCoord[0], mesh.maxCoord[0], ds )[:-1]
allys = (1. - midPlaneDepth)*np.ones(allxs.shape)

midPlaneMasterSwarm = uw.swarm.Swarm( mesh=mesh )
dummy =  midPlaneMasterSwarm.add_particles_with_coordinates(np.column_stack((allxs, allys)))
del allxs
del allys



# In[84]:


#testFn = tm.xFn > -10.
#dummy = pop_or_perish(tm, mCollection, midPlaneMasterSwarm, faultAddFn , ds)
#dummy = remove_faults_from_boundaries(tm, mCollection, testFn )


# In[85]:


#setup a swarm for evaluation surface data on (same points as mesh nodes)

surfacexs = mesh.data[tWalls.data][:,0]
surfaceys = mesh.data[tWalls.data][:,1]
surfLine = usub.interface2D(mesh, velocityField,surfacexs, surfaceys , 0,  2) #
surfVx = uw.swarm.SwarmVariable(surfLine.swarm, 'double', 1)
surfGravStress = uw.swarm.SwarmVariable(surfLine.swarm, 'double', 1)
surfGravTemp = uw.swarm.SwarmVariable(surfLine.swarm, 'double', 1)

surfVx.data[:] = 0
surfGravStress.data[:] = 0
surfGravTemp.data[:] = 0


# ## Gravity

# In[216]:


## Gravity function

#```
#dimensionalisation as follows
G = 6.67e-11         #grav. constant
gravFac = 2.*np.pi*G*1e5 
tempFac = (pd.refDensity*pd.refExpansivity*pd.refLength*(pd.potentialTemp - pd.surfaceTemp)).to_base_units()
stressFac = ((pd.refDiffusivity*pd.refViscosity)/(pd.refGravity*pd.refLength**2)).to_base_units()
#```
#gravFac, tempFac, stressFac 


# In[217]:


## Surface stress
devStressFn =  2.*stokes.fn_viscosity*fn.tensor.symmetric( velocityField.fn_gradient ) #- pressureField
devStressField = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=3 )
projectorDevStress = uw.utils.MeshVariable_Projection( devStressField, devStressFn , type=0 )
projectorDevStress.solve()


meshPressure = uw.mesh.MeshVariable( mesh, 1 )
projectorPressure = uw.utils.MeshVariable_Projection( meshPressure, pressureField, type=0 )
projectorPressure.solve()

#total stress
totalStressFn = -1.*(devStressField[1] - meshPressure)


# In[218]:


from spectral_tools import *

nk = 8
ks = integral_wavenumbers(mesh, nk, axisIndex=1)
upContKernelFn = fn.math.exp(-1.*(1. - fn.coord()[1])*ks)

def update_gravity():
    
    #surface dynamic topography component
    projectorDevStress.solve()
    projectorPressure.solve()
    totalStressFn = -1.*(devStressField[1] - meshPressure)
    surfGravStress.data[:] = totalStressFn.evaluate(surfLine.swarm)
    
    
    synthFn = spectral_integral(mesh, temperatureField, N=nk, axisIndex=1, kernelFn=upContKernelFn, 
                                    average = True, integrationType="volume",surfaceIndexSet=None )
    
    surfGravTemp.data[:] = synthFn.evaluate(surfLine.swarm)


# In[219]:


#update_gravity()


# ## Update functions

# In[220]:


valuesDict = edict({})
valuesDict.timeAtSave = []
valuesDict.stepAtSave = []
for e in tm.undirected.edges():
    valuesDict[str(e)] = []
valuesDict  


# In[221]:


# Here we'll handle everything that should be advected - i.e. every timestep
def advect_update(dt):
    # Retrieve the maximum possible timestep for the advection system.
    #dt = advector.get_max_dt()
    # Advect swarm
    advector.integrate(dt)
    
    #Advect faults
    for f in fCollection:
        f.advection(dt)
        
    #Advect markers
    for m in mCollection:
        m.advection(dt)
    
    
    return time+dt, step+1


# In[222]:


def update_stokes(time, viscosityMapFn ):
    


    
    #set velocity / pressure back to zero
    #velocityField.data[:] = 0.
    #pressureField.data[:] = 0.
    
    #set the new surface vel, get the velXNodes
    vXnodes = set_vel_return_nodes(time, velMaskFn)
    
    #creat a BC object
    velBC = build_velBcs(vXnodes)
    
    
    
    #rebuild stokes
    stokes = uw.systems.Stokes( velocityField  = velocityField, 
                                   pressureField  = pressureField,
                                   conditions     = [velBC,],
                                   fn_viscosity   = viscosityMapFn, 
                                   fn_bodyforce   = buoyancyMapFn )
    return stokes


# In[223]:


def rebuild_solver(stokes):
    
    solver = uw.systems.Solver(stokes)
    solver.set_inner_method("mumps")
    solver.options.scr.ksp_type="cg"
    solver.set_penalty(1.0e7)
    solver.options.scr.ksp_rtol = 1.0e-4
    
    return solver


# In[224]:


#faultloc


# In[225]:


def update_faults():
    
    
    #order is very important here
    dummy = usub.remove_fault_drift(fCollection, faultloc)
    dummy = usub.pop_or_perish(tm, fCollection, faultMasterSwarm, faultAddFn , ds)
    dummy = usub.remove_faults_from_boundaries(tm, fCollection, faultRmfn )
    
    
    for f in fCollection:
        
        #Remove particles below a specified depth
        #depthMask = f.swarm.particleCoordinates.data[:,1] <         (1. - md.faultDestroyDepth)
        depthMask = f.swarm.particleCoordinates.data[:,1] <         (1. - md.wedgeDeep)
        with f.swarm.deform_swarm():
            f.swarm.particleCoordinates.data[depthMask] = (9999999., 9999999.)
        
        #The repair_interface2D routine is supposed to maintain particle density and smooth
        usub.interfaces.repair_interface2D(f, ds, k=8)


# In[226]:


#markerDestroyDepth = ndimlz(500*ur.kilometer)
#markerDestroyDepth


# In[16]:


markerDestroyDepth = ndimlz(500*ur.kilometer)
def update_markers():
        
    #order is very important here
    #dummy = usub.remove_fault_drift(mCollection, 1. - midPlaneDepth)
    dummy = usub.pop_or_perish(tm, mCollection, midPlaneMasterSwarm, faultAddFn , ds)
    dummy = usub.remove_faults_from_boundaries(tm, mCollection, faultRmfn )
    
    
    for f in mCollection:
        
        #Remove particles below a specified depth
        depthMask = f.swarm.particleCoordinates.data[:,1] <         (1. - markerDestroyDepth)
        with f.swarm.deform_swarm():
            f.swarm.particleCoordinates.data[depthMask] = (9999999., 9999999.)
        
        #The repair_interface2D routine is supposed to maintain particle density and smooth
        usub.interfaces.repair_interface2D(f, ds, k=8)
    


# In[15]:


#update_markers()
#usub.interfaces.repair_interface2D
#usub.remove_fault_drift


# In[229]:


def update_swarm():
    
    population_control.repopulate()
    
    for f in fCollection:
        f.rebuild()
        f.set_proximity_director(swarm, proximityVariable, searchFac = 2., locFac=1.0,
                                maxDistanceFn=fn.misc.constant(2.))
        
    #A simple depth cutoff for proximity
    depthMask = swarm.particleCoordinates.data[:,1] < (1. - md.faultDestroyDepth)
    proximityVariable.data[depthMask] = 0
    


# In[6]:


#usub.get_boundary_vel_update


# In[230]:


def update_tect_model(tectModel, tmUwMap, time, dt = 0.0 ):
    
    """
    An example of how we can update the tect_model
    """
    for e in tectModel.undirected.edges():
        
        #This is generally the first condition to check" a specified boundary velocity
        if tectModel.bound_has_vel(e, time):
            newX = usub.get_boundary_vel_update(tectModel, e, time, dt)
            tectModel.set_bound_loc(e, newX)
            
        #in this model the ficticious boundaries remain fixed at the edge
        elif e[0] == e[1]:
            pass       
        #now we'll apply a strain rate query to update the subduction zone loc
        elif tectModel.is_subduction_boundary(e):
            e = tectModel.subduction_edge_order(e)
            newx = usub.strain_rate_field_update(tectModel, e, tmUwMap, dist = ndimlz(100*ur.kilometer))
            tectModel.set_bound_loc(e, newx)
        else:
            pass


# In[231]:


#update_tect_model(tm, tmUwMap, 0., dt = 0.1 )
#usub.strain_rate_field_update


# In[43]:


def update_mask_fns():

    faultRmfn = tm.variable_boundary_mask_fn(distMax = ndimlz(100*ur.km),
                                               distMin = ndimlz(20*ur.km), 
                                               relativeWidth = 0.1,
                                               boundtypes='ridge')



    faultAddFn =  fn.misc.constant(True)

    velMask1 = tm.t2f(tm.ridge_mask_fn(dist=ndimlz(25*ur.km)))
    velMask2= tm.t2f(tm.subduction_mask_fn(dist=md.faultViscHorizTaperStart))
    velMaskFn = operator.and_( velMask1,  velMask2)


    #the following dictates where the fault rheology will be activated
    #subZoneDistfn = tm.subZoneAbsDistFn(upper=True)
    subZoneDistfn = tm.subZoneAbsDistFn(nonSpVal=0.0, upper=False)
    
    faultHorizTaperFn  = usub.cosine_taper(subZoneDistfn, 
                                  md.faultViscHorizTaperStart, md.faultViscHorizTaperWidth)
    
    return faultRmfn, faultAddFn, velMaskFn, faultHorizTaperFn


# In[42]:


#ridgeMaskFn, boundMaskFn, velMaskFn = update_mask_fns()
#usub.cosine_taper
#tm.subZoneAbsDistFn?


# ## Track the values of the plate bounaries

# In[234]:


def update_values():
    
    """ 
    Assumes global variables:
    * time
    * step 
    ...
    + many functions
    """
    
    
    #save the time and step
    valuesDict.timeAtSave.append(time) 
    valuesDict.stepAtSave.append(step)
    
    for e in tm.undirected.edges():
        if tm.is_subduction_boundary(e):
            ee = tm.subduction_edge_order(e) #hacky workaround for the directed/ undireted. need get_bound_loc
        else:
            ee = e

        valuesDict[str(e)].append(tm.get_bound_loc(ee))
        
        
    #save
    if uw.rank()==0:
        fullpath = os.path.join(valuesPath + "tect_model_data")
        #the '**' is important
        np.savez(fullpath, **valuesDict)
    


# In[136]:


#sigXXswarm =  2.*symStrainrate[0]*viscosityMapFn
viscMesh = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
sigXXMesh =  uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
viscMesh.data[:] = 0.
sigXXMesh.data[:] = 0.

sigXXFn = 2.*symStrainrate[0]*sigXXMesh
#projectorDevStress = uw.utils.MeshVariable_Projection(sigXXMesh, sigXXswarm  , type=0 )

def swarm_to_mesh_update():    
#    projectorDevStress.solve()
    ix1, weights1, d1 = nn_evaluation(swarm.particleCoordinates.data, mesh.data, n=5, weighted=True)
    viscMesh.data[:,0] =  np.average(viscosityMapFn.evaluate(swarm)[:,0][ix1], weights=weights1, axis=len((weights1.shape)) - 1)
    #sigXXFn may need updating?
    sigXXFn = 2.*symStrainrate[0]*viscMesh
    sigXXMesh.data[:] = sigXXFn.evaluate(mesh)


# In[137]:


#sigXXFn = 2.*symStrainrate[0]*viscMesh
#sigXXMesh.data[:] = sigXXFn.evaluate(mesh)


# In[138]:


#step = 0
#files_update()
swarm_to_mesh_update()


# In[239]:


#fig = glucifer.Figure(figsize=(960,300) )
#fig.append( glucifer.objects.Surface(mesh, sigXXMesh, onMesh=True))
#fig.show()


# In[154]:


parallelMeshVariable = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
sigSSMesh = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )

lithHalfThick = ndimlz(25.*ur.kilometer)


def update_slab_parallel_sr():
    
    #update the orientation mesh var
    parallelMeshVariable.data[:] = 0.
    director, fpts = mCollection[0].compute_normals(mesh.data, thickness=lithHalfThick)
    parDir = np.column_stack((-1.*director[:,1], director[:,0]))
    parallelMeshVariable.data[fpts] = parDir[fpts]
    
    #compute the 2X2 strain rate structure
    strainRateData =  symStrainrate.evaluate(mesh)
    fullStrainRate = np.zeros((mesh.data.shape[0], 2,2))
    fullStrainRate[:,0,0] = strainRateData[:,0]
    fullStrainRate[:,1,1] = strainRateData[:,1]
    fullStrainRate[:,0,1] = strainRateData[:,2]
    fullStrainRate[:,1,0] = strainRateData[:,2]
    
    #resolve the strain rate
    traction =  np.matmul(fullStrainRate,parallelMeshVariable.data[..., np.newaxis])[:,:,0]
    resolvedSR = np.einsum('ij,ij->i', traction , parallelMeshVariable.data)
    #print(resolvedSR.shape)
    
    sigSSMesh.data[:,0] = 2.*resolvedSR*viscMesh.data[:,0]
    #mask values above the maxYieldStress
    mask = np.abs(sigSSMesh.data[:,0] ) > ndp.yieldStressMax
    sigSSMesh.data[:,0][mask] = ndp.yieldStressMax


# In[175]:


#update_slab_parallel_sr()
#mask = np.abs(sigSSMesh.data[:,0] ) > ndp.yieldStressMax
#sigSSMesh.data[:,0][mask] = ndp.yieldStressMax
#params.stressScale


# In[240]:


def files_update():
    fCollection[0].swarm.save( filePath + "interface" + str(step).zfill(5))
    mCollection[0].swarm.save( filePath + "midplane" + str(step).zfill(5))
    
    surfVx.data[:] = velocityField[0].evaluate(surfLine.swarm)
    surfVx.save(filePath + "surfVx_" + str(step).zfill(3) + "_.h5")
    
    surfGravStress.save(filePath + "gravStress_" + str(step).zfill(3) + "_.h5")
    surfGravTemp.save(filePath + "gravTemp_" + str(step).zfill(3) + "_.h5")


# In[241]:


def xdmfs_update():
 
    try:
        _mH
    except:
        _mH = mesh.save(xdmfPath+"mesh.h5")
    
    #part1
    mh = _mH
    tH = temperatureFn.save(xdmfPath + "temp_" + str(step) + ".h5")
    pH = pressureField.save(xdmfPath + "press_" + str(step) + ".h5")
    visc = viscMesh.save(xdmfPath + "visc_" + str(step) + ".h5")
    sigXX = sigXXMesh.save(xdmfPath + "sigXX_" + str(step) + ".h5")
    sigSS = sigSSMesh.save(xdmfPath + "sigSS_" + str(step) + ".h5")
    
    #part 2
    temperatureFn.xdmf(xdmfPath + "temp_" + str(step), tH, 'temperature', mh, 'mesh', modeltime=time)
    pressureField.xdmf(xdmfPath + "press_" + str(step), pH, 'pressure', mh, 'mesh', modeltime=time)
    sigXXMesh.xdmf(xdmfPath+ "sigXX_" + str(step), sigXX, 'sigXX', mh, 'mesh', modeltime=time)
    sigSSMesh.xdmf(xdmfPath+ "sigSS_" + str(step), sigSS, 'sigSS', mh, 'mesh', modeltime=time)
    viscMesh.xdmf(xdmfPath+ "visc_" + str(step), visc , 'visc', mh, 'mesh', modeltime=time)


# In[242]:


#step = 99
#time = 0
#xdmfs_update()
#viscMesh.evaluate(mesh).min()


# ## Viz

# In[243]:


viscSwarmVar =  swarm.add_variable( dataType="double", count=1 )
viscSwarmVar.data[:] = viscosityMapFn.evaluate(swarm)


maskFnVar1 = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=1 )
maskFnVar1.data[:] = velMaskFn.evaluate(mesh)


maskFnVar2 = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=1 )
maskFnVar2.data[:] = faultRmfn.evaluate(mesh)

plate_id_fn = tm.plate_id_fn()
maskFnVar3 = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=1 )
maskFnVar3.data[:] = plate_id_fn.evaluate(mesh)


# In[244]:


store1 = glucifer.Store(dbPath+ 'subduction1.gldb')
store2 = glucifer.Store(dbPath + 'subduction2.gldb')
store3 = glucifer.Store(dbPath+ 'subduction3.gldb')



figTemp = glucifer.Figure(store1, figsize=(960,300) )
figTemp.append( glucifer.objects.Surface(mesh, temperatureField, onMesh=True))
figTemp.append( glucifer.objects.Contours(mesh, temperatureField, resolution=300, quality =4, interval=0.2,  colours='Black', colourBar=False)) 
for f in fCollection:
    figTemp.append( glucifer.objects.Points(f.swarm, pointSize=5))
for m in mCollection:
    figTemp.append( glucifer.objects.Points(m.swarm, pointSize=5))
    
#figProx = glucifer.Figure(store1, figsize=(960,300) )
#figProx.append( glucifer.objects.Points(swarm , proximityVariable))
#for f in fCollection:
#    figProx.append( glucifer.objects.Points(f.swarm, pointSize=5))
#figProx.show()

figVisc = glucifer.Figure( store2, figsize=(960,300) )
#figVisc.append( glucifer.objects.Points(swarm, viscosityMapFn, pointSize=2, logScale=True) )
figVisc.append( glucifer.objects.Points(swarm,  viscSwarmVar, logScale=True) )
#figVisc.append( glucifer.objects.Points(swarm,  wedgeVariable) )



#figVel = glucifer.Figure( store3, figsize=(960,300) )
#figVel.append(glucifer.objects.Surface(mesh, fn.math.dot(velocityField, velocityField), onMesh=True))
#figVel.append( glucifer.objects.VectorArrows(mesh, velocityField, arrowHead=0.2, scaling=1./refVel) )

figMask = glucifer.Figure( store3, figsize=(960,300) )
figMask.append( glucifer.objects.Surface(mesh,  maskFnVar1) )
figMask.append( glucifer.objects.Surface(mesh,  maskFnVar2) )
figMask.append( glucifer.objects.Surface(mesh, maskFnVar3 , valueRange=[0,3]) )
for f in fCollection:
    figMask.append( glucifer.objects.Points(f.swarm, pointSize=5))


#figMask.append( glucifer.objects.Surface(mesh,  maskFnVar3) )
#for f in fCollection:
#    figVel.append( glucifer.objects.Points(f.swarm, pointSize=5))



# ## Main Loop

# In[ ]:


time = 0.  # Initial time
step = 0 
maxSteps = 2000      # Maximum timesteps 
steps_output = 5   # output every N timesteps
swarm_update = 10   # output every N timesteps
faults_update = 10
dt_model = 0.
steps_update_model = 5

wedgeOn = False
update_values()


# In[ ]:


while time < tm.times[-1] and step < maxSteps:
    
    # Solve non linear Stokes system
    solver.solve(nonLinearIterate=True, nonLinearTolerance=md.nltol, callback_post_solve = pressure_calibrate)
    
    dt = advDiff.get_max_dt()
    advDiff.integrate(dt)
    
    #advect swarm and faults
    time, step =  advect_update(dt)
    dt_model += dt
    
        
    #update tectonic model
    if step % steps_update_model == 0:
        update_tect_model(tm, tmUwMap, time, dt = dt_model)
        dt_model = 0.
        plate_id_fn = tm.plate_id_fn()
        faultRmfn, faultAddFn, velMaskFn, faultHorizTaperFn = update_mask_fns()
        
        #these need to be explicity updated
        faultRheologyFn =  faultViscosityFn*(1. - faultDepthTaperFn) +        faultDepthTaperFn*mantleRheologyFn + faultHorizTaperFn*mantleRheologyFn
        
        viscosityMapFn = fn.branching.map( fn_key = proximityVariable,
                             mapping = {0:mantleRheologyFn_,
                                        2:faultRheologyFn} )
        
        
        
    #running fault healing/addition, map back to swarm
    if step % faults_update == 0:
        update_faults()
        update_markers()
    if step % swarm_update == 0:
        update_swarm()
        
    #rebuild stokes
    if step % steps_update_model == 0:
        del solver
        del stokes
        stokes = update_stokes(time, viscosityMapFn )
        solver = rebuild_solver(stokes)
        
    
    # output figure to file at intervals = steps_output
    if step % steps_output == 0 or step == maxSteps-1:
        
        update_values()
        
        #Important to set the timestep for the store object here or will overwrite previous step
        
        #also update this guy for viz
        viscSwarmVar.data[:] = viscosityMapFn.evaluate(swarm)
        maskFnVar1.data[:] = velMaskFn.evaluate(mesh)
        maskFnVar2.data[:] = faultRmfn.evaluate(mesh)
        maskFnVar3.data[:] = plate_id_fn.evaluate(mesh)
        
        store1.step = step
        store2.step = step
        store3.step = step
        figTemp.save(    outputPath + "temp"    + str(step).zfill(4))
        #figProx.save(    outputPath + "prox"    + str(step).zfill(4))
        figVisc.save(    outputPath + "visc"    + str(step).zfill(4))
        figMask.save(    outputPath + "mask"    + str(step).zfill(4))
        #figVel.save(    outputPath + "vel"    + str(step).zfill(4))
        
        #save out the files
        
        update_gravity()
        files_update()
        
        #XDMFS
        swarm_to_mesh_update()
        update_slab_parallel_sr()
        xdmfs_update()
        
    ##Wedge stuff 
    
    
    if time > md.turnOnWedge:
        
        faultDepthTaperFn = usub.cosine_taper(depthFn,
                                 0.75*md.faultViscDepthTaperStart, md.faultViscDepthTaperWidth)

        
        #toggle the rheology. 
        #this should only happen once, hence the wedgeOn var. 
        if wedgeOn == False:
            mantleRheologyFn_ = fn.branching.map( fn_key = wedgeVariable,
                                         mapping = {0:mantleRheologyFn,
                                                    1:md.wedgeViscosity} )
       
            viscosityMapFn = fn.branching.map( fn_key = proximityVariable,
                             mapping = {0:mantleRheologyFn_,
                                        2:faultRheologyFn} )
           
            stokes.fn_viscosity = viscosityMapFn 
            
            wedgeOn = True
        
        #Now update the wedge region
        build_wedge_region()


    print 'wedge on is: {}'.format(wedgeOn)    
    
    
    if uw.rank()==0:
        print 'step = {0:6d}; time = {1:.3e}'.format(step,time)


# In[199]:


#stokes.fn_viscosity = viscosityMapFn


# In[163]:





# In[88]:


#tm.is_ridge((3,3))

