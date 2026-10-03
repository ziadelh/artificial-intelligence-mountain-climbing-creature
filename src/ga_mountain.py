import genome
import creature
import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import random
import argparse
import csv
import matplotlib
matplotlib.use("Agg")   # draw the charts to files, so the script also runs without a display
import matplotlib.pyplot as plt
import math
import pandas as pd


def setup_world():
    p.resetSimulation()
    p.setGravity(0, 0, -5)
    arena_size = 20
    wall_thickness = 0.5
    wall_height = 5
    floor_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness])
    floor_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[arena_size/2, arena_size/2, wall_thickness], rgbaColor=[1, 1, 0, 1])
    ground_id = p.createMultiBody(0, floor_collision_shape, floor_visual_shape, [0, 0, -wall_thickness])
    mountain_id = p.loadURDF("gaussian_pyramid.urdf", [0, 0, -1], useFixedBase=1)
    # the mountain mesh has no material, so give it a colour (otherwise the GUI draws it black)
    p.changeVisualShape(mountain_id, -1, rgbaColor=[0.55, 0.45, 0.35, 1])
    wall_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2])
    wall_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2], rgbaColor=[0.2, 0.2, 0.2, 1])
    positions = [
        ([0, arena_size / 2, wall_height / 2]),
        ([0, -arena_size / 2, wall_height / 2]),
    ]
    for pos in positions:
        p.createMultiBody(0, wall_col, wall_vis, pos)
    wall_col_side = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2])
    wall_vis_side = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2], rgbaColor=[0.2, 0.2, 0.2, 1])
    positions = [
        ([arena_size / 2, 0, wall_height / 2]),
        ([-arena_size / 2, 0, wall_height / 2]),
    ]
    for pos in positions:
        p.createMultiBody(0, wall_col_side, wall_vis_side, pos)
    p.changeDynamics(ground_id, -1, lateralFriction=0.2, spinningFriction=0.1, rollingFriction=0.1)


def evaluate_creature(creature_obj, steps=240 * 120):
    """Simulate one creature on the mountain and return its fitness.

    steps is the number of physics steps (240 steps = 1 simulated second).
    """
    setup_world()
    p.setGravity(0, 0, -4.0)
    urdf_str = creature_obj.to_xml()
    with open("test.urdf", "w") as f:
        f.write(urdf_str)
    try:
        robot_id = p.loadURDF("test.urdf", [0, 0, 10])
        motors = creature_obj.motors
        num_joints = p.getNumJoints(robot_id)
        for j in range(num_joints):
            p.changeDynamics(robot_id, j, jointLowerLimit=-np.pi, jointUpperLimit=np.pi, linearDamping=0.02, angularDamping=0.02)
            p.changeDynamics(robot_id, j, lateralFriction=0.5, spinningFriction=0.05, rollingFriction=0.07)
        p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        motor_joint_map = {j: motors[j % len(motors)] for j in range(num_joints)}
        heights = []
        positions = []
        joint_velocities = []
        for i in range(steps):
            for j in range(num_joints):
                try:
                    motor = motor_joint_map[j]
                    motor.amp = min(motor.amp, 0.5)
                    motor.freq = min(motor.freq, 0.2)
                    motor_output = motor.get_output()
                    p.setJointMotorControl2(
                        bodyIndex=robot_id,
                        jointIndex=j,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=motor_output,
                        force=500
                    )
                except Exception:
                    p.removeBody(robot_id)
                    return 0.0
            try:
                p.stepSimulation()
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                joint_states = p.getJointStates(robot_id, range(num_joints))
                velocities = [js[1] for js in joint_states]
                joint_velocities.append(np.mean(np.abs(velocities)))
            except Exception:
                p.removeBody(robot_id)
                return 0.0
            z = pos[2]
            heights.append(z)
            positions.append(pos)
        if not heights:
            p.removeBody(robot_id)
            return 0.0
        start_y = positions[0][1]
        end_y = positions[-1][1]
        start_z = heights[0]
        end_z = heights[-1]
        max_z = max(heights)
        ground_z = min(heights[:20])
        vertical_gain = max_z - ground_z
        forward_progress = max(end_y - start_y, 0)

        steps_above_base = sum(1 for z in heights if z > ground_z + 0.05)
        fraction_above_base = steps_above_base / len(heights)

        fitness = 5.0 * forward_progress + 40.0 * vertical_gain

        if vertical_gain > 0.05:
            fitness += 50.0

        if fraction_above_base > 0.1:
            fitness += 30.0

        if end_y > 0.5:
            fitness += 10.0

        if vertical_gain < 0.05 and forward_progress < 0.05:
            fitness -= 5.0

        vibration_penalty = -0.2 * np.std(heights) - 0.1 * np.mean(joint_velocities)
        fitness += vibration_penalty

        if abs(positions[-1][0]) > 8.0 or abs(positions[-1][1]) > 8.0:
            fitness -= 10.0

        p.removeBody(robot_id)
        return fitness
    except Exception as e:
        return 0.0

def tournament_selection(fitness_scores, k=5):
    selected = random.sample(fitness_scores, k)
    selected.sort(key=lambda x: x[0], reverse=True)
    return selected[0][1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Mountain Climbing Creatures")
    parser.add_argument('--pop_size', type=int, default=40, help='Population size')
    parser.add_argument('--generations', type=int, default=50, help='Number of generations')
    parser.add_argument('--logfile', type=str, default='fitness_log.csv', help='CSV file to log fitness results')
    parser.add_argument('--steps', type=int, default=240 * 120, help='Physics steps per creature (240 steps = 1 simulated second)')
    parser.add_argument('--gui', action='store_true', help='Show the PyBullet window (slower)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed, to repeat a run')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "shapes"))
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0/240)

    POP_SIZE = args.pop_size
    GENERATIONS = args.generations
    LOGFILE = args.logfile

    population = []
    for _ in range(8):
        climber_dna = genome.Genome.get_good_seed_genome(len(genome.Genome.get_gene_spec()))
        for gene in climber_dna:
            gene[genome.Genome.get_gene_spec()["link-length"]["ind"]] = np.random.uniform(1.1, 1.3)
            gene[genome.Genome.get_gene_spec()["control-amp"]["ind"]] = np.random.uniform(0.6, 0.8)
            gene[genome.Genome.get_gene_spec()["control-freq"]["ind"]] = np.random.uniform(0.18, 0.28)
        while len(climber_dna) < 12:
            climber_dna.append(genome.Genome.get_random_gene(len(genome.Genome.get_gene_spec()), chain_init=True))
        population.append(creature.Creature.from_genome(climber_dna))
    for _ in range(POP_SIZE - 8):
        rand_dna = genome.Genome.get_random_creature(len(genome.Genome.get_gene_spec()))
        population.append(creature.Creature.from_genome(rand_dna))

    fitness_history = []
    climber_table = []

    with open(LOGFILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Generation', 'BestFitness', 'AverageFitness'])

        for gen in range(GENERATIONS):
            print(f"\n Generation {gen}")
            fitness_scores = []
            climber_count = 0
            top_climber_count = 0
            for i, cr in enumerate(population):
                fitness = evaluate_creature(cr, steps=args.steps)
                fitness_scores.append((fitness, cr))
                print(f"  Creature {i}: Fitness = {fitness:.3f}")

                if fitness >= 50:
                    climber_count += 1
                if fitness >= 200:
                    top_climber_count += 1

            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            best_fitness = fitness_scores[0][0]
            avg_fitness = sum(f[0] for f in fitness_scores) / len(fitness_scores)
            print(f" Best Fitness This Gen: {best_fitness:.3f} | Avg: {avg_fitness:.3f}")

            writer.writerow([gen, best_fitness, avg_fitness])
            fitness_history.append((best_fitness, avg_fitness))
            climber_table.append({
                'Generation': gen,
                'BestFitness': best_fitness,
                'AvgFitness': avg_fitness,
                'Climbers': climber_count,
                'TopClimbers': top_climber_count
            })

            num_elites = min(2, len(fitness_scores))
            new_population = [fitness_scores[i][1] for i in range(num_elites)]

            while len(new_population) < POP_SIZE - 2:
                parent1 = tournament_selection(fitness_scores, k=5)
                parent2 = tournament_selection(fitness_scores, k=5)
                dna1 = parent1.dna
                dna2 = parent2.dna
                child_dna = genome.Genome.crossover(dna1, dna2)
                child_dna = genome.Genome.point_mutate(child_dna, rate=0.9, amount=0.4)
                child_dna = genome.Genome.grow_mutate(child_dna, rate=0.1)
                child_dna = genome.Genome.shrink_mutate(child_dna, rate=0.1)
                if isinstance(child_dna, np.ndarray):
                    if len(child_dna.shape) == 2:
                        child_dna = [child_dna[i, :] for i in range(child_dna.shape[0])]
                    elif len(child_dna.shape) == 1:
                        child_dna = [child_dna]
                new_population.append(creature.Creature.from_genome(child_dna))

            for _ in range(2):
                rand_dna = genome.Genome.get_random_creature(len(genome.Genome.get_gene_spec()))
                new_population.append(creature.Creature.from_genome(rand_dna))

            population = new_population

    df = pd.DataFrame(climber_table)
    df.to_csv('climber_table.csv', index=False)
    print("Saved climber_table.csv")
    generations = list(range(GENERATIONS))
    best = [x[0] for x in fitness_history]
    avg = [x[1] for x in fitness_history]
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best, label='Best Fitness')
    plt.plot(generations, avg, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progress Over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fitness_progress.png')
    print("Saved fitness_progress.png")
    plt.figure(figsize=(10, 6))
    plt.plot(generations, [row['Climbers'] for row in climber_table], label='Climbers (z > base)')
    plt.plot(generations, [row['TopClimbers'] for row in climber_table], label='Top Climbers (z > top)')
    plt.xlabel('Generation')
    plt.ylabel('Number of Climbers')
    plt.title('Number of Climbers per Generation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('climber_counts.png')
    print("Saved climber_counts.png")

    p.disconnect()
