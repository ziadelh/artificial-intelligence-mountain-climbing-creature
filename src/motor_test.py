import genome
import creature
import pybullet as p
import time 
import random
import numpy as np

## ... usual starter code to create a sim and floor
p.connect(p.GUI)
p.setPhysicsEngineParameter(enableFileCaching=0)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
plane_shape = p.createCollisionShape(p.GEOM_PLANE)
floor = p.createMultiBody(plane_shape, plane_shape)
p.setGravity(0, 0, -10)


# generate a random creature
cr = creature.Creature(gene_count=3)
with open('test.urdf', 'w') as f:
    f.write(cr.to_xml())
rob1 = p.loadURDF('test.urdf')
start_pos, orn = p.getBasePositionAndOrientation(rob1)

# iterate 
elapsed_time = 0
wait_time = 1.0 / 240 
total_time = 5 
step = 0
dist_moved = 0
while True:
    p.stepSimulation()
    step += 1
    if step % 120 == 0:
        motors = cr.get_motors()
        assert len(motors) == p.getNumJoints(rob1), "Something went wrong"
        for jid in range(p.getNumJoints(rob1)):
            mode = p.VELOCITY_CONTROL
            vel = motors[jid].get_output()
            p.setJointMotorControl2(rob1, 
                        jid,  
                        controlMode=mode, 
                        targetVelocity=vel)
        new_pos, orn = p.getBasePositionAndOrientation(rob1)
        dist_moved = np.linalg.norm(np.asarray(start_pos) - np.asarray(new_pos))
    print(dist_moved)
    time.sleep(wait_time)
    elapsed_time += wait_time
    if elapsed_time > total_time:
        break

print("TOTAL DISTANCE MOVED:", dist_moved)

