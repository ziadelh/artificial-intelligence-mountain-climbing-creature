import numpy as np
import copy
import random

class Genome():
    @staticmethod
    def get_random_gene(length, link_index=None, chain_init=False):
        gene = np.zeros(length)
        spec = Genome.get_gene_spec()
        for key, val in spec.items():
            ind = val["ind"]
            if chain_init:
                if link_index == 0:
                    if key == "link-length":
                        gene[ind] = np.random.uniform(0.4, 1.0)
                    elif key == "link-radius":
                        gene[ind] = np.random.uniform(0.2, 0.5)
                    elif key == "link-mass":
                        gene[ind] = np.random.uniform(0.005, 0.03)
                    elif key == "joint-parent":
                        gene[ind] = 0.0
                    elif key.startswith("joint-origin-xyz") or key == "joint-axis-xyz":
                        gene[ind] = np.random.uniform(-1, 1)
                    elif key == "joint-origin-rpy-2":
                        gene[ind] = np.random.uniform(1.0, 2.0)
                    elif key == "joint-axis-xyz":
                        gene[ind] = np.random.uniform(0.5, 1.0)
                    else:
                        gene[ind] = np.random.random()
                else:
                    if key == "link-length":
                        gene[ind] = np.random.uniform(0.4, 1.2)
                    elif key == "link-radius":
                        gene[ind] = np.random.uniform(0.15, 0.4)
                    elif key == "link-mass":
                        gene[ind] = np.random.uniform(0.01, 0.2)
                    elif key == "control-amp":
                        gene[ind] = np.random.uniform(0.2, 0.7)
                    elif key == "control-freq":
                        gene[ind] = np.random.uniform(0.05, 0.3)
                    elif key == "control-phase":
                        if link_index is not None and link_index % 2 == 0:
                            gene[ind] = 0.0
                        else:
                            gene[ind] = 0.5
                    elif key == "control-waveform":
                        gene[ind] = np.random.uniform(0.7, 1.0)
                    elif key == "joint-parent" and link_index is not None and link_index > 0:
                        parent_idx = np.random.randint(0, link_index)
                        gene[ind] = parent_idx / link_index
                    elif key.startswith("joint-origin-xyz") or key == "joint-axis-xyz":
                        gene[ind] = np.random.uniform(-1, 1)
                    elif key == "joint-origin-rpy-1":
                        gene[ind] = np.random.uniform(0.0, 1.0)
                    elif key == "joint-origin-rpy-3":
                        gene[ind] = np.random.uniform(0.0, 1.0)
                    elif key == "link-recurrence":
                        gene[ind] = 0
                    elif key == "joint-origin-rpy-2":
                        gene[ind] = np.random.uniform(1.0, 2.0)
                    elif key == "joint-axis-xyz":
                        gene[ind] = np.random.uniform(0.5, 1.0)
                    else:
                        gene[ind] = np.random.random()
            else:
                gene[ind] = np.random.random()
        return gene
    @staticmethod
    def get_random_genome(gene_length, gene_count, chain_init=False):
        gene_count = max(gene_count, 6)
        gene_count = np.random.randint(6, 13)
        genome = [Genome.get_random_gene(gene_length, i, chain_init=chain_init) for i in range(gene_count)]
        if chain_init and gene_count > 1:
            for i, gene in enumerate(genome):
                phase_ind = Genome.get_gene_spec()["control-phase"]["ind"]
                gene[phase_ind] = 0.0 if i % 2 == 0 else 0.5
                if i < 3:
                    angle_ind = Genome.get_gene_spec()["joint-origin-rpy-1"]["ind"]
                    gene[angle_ind] = 0.7 + (i * 0.1)
                    axis_ind = Genome.get_gene_spec()["joint-axis-xyz"]["ind"]
                    gene[axis_ind] = 0.7 + (i * 0.2)
        return genome
    @staticmethod
    def get_gene_spec():
        gene_spec =  {"link-shape":{"scale":1},
            "link-length": {"scale":1.0},
            "link-radius": {"scale":0.2},
            "link-recurrence": {"scale":3},
            "link-mass": {"scale":1},
            "joint-type": {"scale":1},
            "joint-parent":{"scale":1},
            "joint-axis-xyz": {"scale":1},
            "joint-origin-rpy-1":{"scale":np.pi * 2},
            "joint-origin-rpy-2":{"scale":np.pi * 2},
            "joint-origin-rpy-3":{"scale":np.pi * 2},
            "joint-origin-xyz-1":{"scale":1},
            "joint-origin-xyz-2":{"scale":1},
            "joint-origin-xyz-3":{"scale":1},
            "control-waveform":{"scale":1},
            "control-amp":{"scale":1.0},
            "control-freq":{"scale":1},
            "control-phase":{"scale":np.pi * 2}
            }
        ind = 0
        for key in gene_spec.keys():
            gene_spec[key]["ind"] = ind
            ind = ind + 1
        return gene_spec
    @staticmethod
    def get_gene_dict(gene, spec):
        gdict = {}
        for key in spec:
            ind = spec[key]["ind"]
            scale = spec[key]["scale"]
            gdict[key] = gene[ind] * scale
        return gdict
    @staticmethod
    def get_genome_dicts(genome, spec):
        gdicts = []
        for gene in genome:
            gdicts.append(Genome.get_gene_dict(gene, spec))
        return gdicts
    @staticmethod
    def parent_index_from_gene(parent_gene, link_index):
        if link_index <= 0:
            return None
        return int(parent_gene * link_index)
    @staticmethod
    def genome_to_links(gdicts):
        links = []
        for i, gdict in enumerate(gdicts):
            link_name = str(i)
            if i == 0:
                parent_name = "None"
            else:
                parent_idx = Genome.parent_index_from_gene(gdict["joint-parent"], i)
                parent_name = str(parent_idx)
            link = URDFLink(name=link_name,
                            parent_name=parent_name,
                            recur=1,
                            link_length=gdict["link-length"],
                            link_radius=gdict["link-radius"],
                            link_mass=gdict["link-mass"],
                            joint_type=gdict["joint-type"],
                            joint_parent=gdict["joint-parent"],
                            joint_axis_xyz=gdict["joint-axis-xyz"],
                            joint_origin_rpy_1=gdict["joint-origin-rpy-1"],
                            joint_origin_rpy_2=gdict["joint-origin-rpy-2"],
                            joint_origin_rpy_3=gdict["joint-origin-rpy-3"],
                            joint_origin_xyz_1=gdict["joint-origin-xyz-1"],
                            joint_origin_xyz_2=gdict["joint-origin-xyz-2"],
                            joint_origin_xyz_3=gdict["joint-origin-xyz-3"],
                            control_waveform=gdict["control-waveform"],
                            control_amp=gdict["control-amp"],
                            control_freq=gdict["control-freq"],
                            control_phase=gdict["control-phase"])
            links.append(link)
        return links
    @staticmethod
    def expandLinks(parent_link, uniq_parent_name, flat_links, exp_links):
        return
    @staticmethod
    def crossover(g1, g2):
        min_links = 2
        x1 = random.randint(0, len(g1)-1)
        x2 = random.randint(0, len(g2)-1)
        g3 = np.concatenate((g1[x1:], g2[x2:]))
        if len(g3) < min_links:
            gene_length = len(g1[0])
            while len(g3) < min_links:
                g3 = np.append(g3, [Genome.get_random_gene(gene_length)], axis=0)
        if len(g3) > len(g1):
            g3 = g3[0:len(g1)]
        return [np.array(g) for g in g3]
    @staticmethod
    def point_mutate(genome, rate, amount):
        new_genome = copy.copy(genome)
        for gene_idx, gene in enumerate(new_genome):
            for i in range(len(gene)):
                if random.random() < 0.8:
                    if i in [Genome.get_gene_spec()[f"joint-origin-xyz-{j}"]["ind"] for j in [1,2,3]] or i == Genome.get_gene_spec()["joint-axis-xyz"]["ind"]:
                        gene[i] += random.uniform(-0.5, 0.5)
                        gene[i] = max(-1.0, min(1.0, gene[i]))
                    else:
                        gene[i] += random.uniform(-0.5, 0.5)
                        gene[i] = min(max(gene[i], 0.0), 0.9999)
            amp_ind = Genome.get_gene_spec()["control-amp"]["ind"]
            freq_ind = Genome.get_gene_spec()["control-freq"]["ind"]
            gene[amp_ind] = max(gene[amp_ind], 0.15)
            gene[freq_ind] = max(gene[freq_ind], 0.08)
            phase_ind = Genome.get_gene_spec()["control-phase"]["ind"]
            gene[phase_ind] = random.uniform(0, 1)
            parent_ind = Genome.get_gene_spec()["joint-parent"]["ind"]
            if random.random() < 0.2 and gene_idx != 0:
                link_index = gene_idx
                if link_index > 0:
                    parent_idx = np.random.randint(0, link_index)
                    gene[parent_ind] = parent_idx / link_index
        return new_genome
    @staticmethod
    def shrink_mutate(genome, rate):
        if len(genome) <= 2:
            return [np.array(g) for g in genome]
        if random.random() < rate:
            ind = random.randint(0, len(genome)-1)
            new_genome = np.delete(genome, ind, 0)
            if len(new_genome) < 2:
                return [np.array(g) for g in genome]
            return [np.array(g) for g in new_genome]
        else:
            return [np.array(g) for g in genome]
    @staticmethod
    def grow_mutate(genome, rate):
        if random.random() < rate:
            gene = Genome.get_random_gene(len(genome[0]))
            new_genome = copy.copy(genome)
            new_genome = np.append(new_genome, [gene], axis=0)
            return [np.array(g) for g in new_genome]
        else:
            return [np.array(g) for g in genome]
    @staticmethod
    def count_base_links(genome, spec):
        count = 0
        for i, gene in enumerate(genome):
            if i == 0:
                continue
            parent_val = gene[spec["joint-parent"]["ind"]]
            if parent_val < 0.01:
                count += 1
        return count
    @staticmethod
    def to_csv(dna, csv_file):
        csv_str = ""
        for gene in dna:
            for val in gene:
                csv_str = csv_str + str(val) + ","
            csv_str = csv_str + '\n'

        with open(csv_file, 'w') as f:
            f.write(csv_str)

    @staticmethod
    def from_csv(filename):
        csv_str = ''
        with open(filename) as f:
            csv_str = f.read()   
        dna = []
        lines = csv_str.split('\n')
        for line in lines:
            vals = line.split(',')
            gene = [float(v) for v in vals if v != '']
            if len(gene) > 0:
                dna.append(gene)
        return dna

    @staticmethod
    def get_good_seed_genome(gene_length):
        genome = []
        for i in range(8):
            gene = Genome.get_random_gene(gene_length, i, chain_init=True)
            if i % 2 == 0:
                gene[Genome.get_gene_spec()["control-phase"]["ind"]] = 0.0
            else:
                gene[Genome.get_gene_spec()["control-phase"]["ind"]] = 0.5
            gene[Genome.get_gene_spec()["link-length"]["ind"]] = np.random.uniform(0.6, 1.2)  
            gene[Genome.get_gene_spec()["joint-origin-rpy-2"]["ind"]] = np.random.uniform(1.0, 2.0)  
            gene[Genome.get_gene_spec()["joint-axis-xyz"]["ind"]] = np.random.uniform(0.5, 1.0)  
            gene[Genome.get_gene_spec()["link-mass"]["ind"]] = np.random.uniform(0.01, 0.05)
            genome.append(gene)
        return genome

    @staticmethod
    def get_random_creature(gene_length):
        gene_count = np.random.randint(6, 13)
        genome = [Genome.get_random_gene(gene_length, i, chain_init=True) for i in range(gene_count)]
        return genome

class URDFLink:
    def __init__(self, name, parent_name, recur, 
                link_length=0.1, 
                link_radius=0.1, 
                link_mass=0.1,
                joint_type=0.1,
                joint_parent=0.1,
                joint_axis_xyz=0.1,
                joint_origin_rpy_1=0.1,
                joint_origin_rpy_2=0.1,
                joint_origin_rpy_3=0.1,
                joint_origin_xyz_1=0.1,
                joint_origin_xyz_2=0.1,
                joint_origin_xyz_3=0.1,
                control_waveform=0.1,
                control_amp=0.1,
                control_freq=0.1,
                control_phase=0.0):
        self.name = name
        self.parent_name = parent_name
        self.recur = recur 
        self.link_length=link_length 
        self.link_radius=link_radius
        self.link_mass=link_mass
        self.joint_type=joint_type
        self.joint_parent=joint_parent
        self.joint_axis_xyz=joint_axis_xyz
        self.joint_origin_rpy_1=joint_origin_rpy_1
        self.joint_origin_rpy_2=joint_origin_rpy_2
        self.joint_origin_rpy_3=joint_origin_rpy_3
        self.joint_origin_xyz_1=joint_origin_xyz_1
        self.joint_origin_xyz_2=joint_origin_xyz_2
        self.joint_origin_xyz_3=joint_origin_xyz_3
        self.control_waveform=control_waveform
        self.control_amp=control_amp
        self.control_freq=control_freq
        self.control_phase=control_phase
        self.sibling_ind = 1

    def to_link_element(self, adom):
        link_tag = adom.createElement("link")
        link_tag.setAttribute("name", self.name)
        vis_tag = adom.createElement("visual")
        geom_tag = adom.createElement("geometry")
        cyl_tag = adom.createElement("cylinder")
        cyl_tag.setAttribute("length", str(self.link_length))
        cyl_tag.setAttribute("radius", str(self.link_radius))
        geom_tag.appendChild(cyl_tag)
        vis_tag.appendChild(geom_tag)
        coll_tag = adom.createElement("collision")
        c_geom_tag = adom.createElement("geometry")
        c_cyl_tag = adom.createElement("cylinder")
        c_cyl_tag.setAttribute("length", str(self.link_length))
        c_cyl_tag.setAttribute("radius", str(self.link_radius))
        c_geom_tag.appendChild(c_cyl_tag)
        coll_tag.appendChild(c_geom_tag)
        inertial_tag = adom.createElement("inertial")
        mass_tag = adom.createElement("mass")
        mass_tag.setAttribute("value", str(self.link_mass))
        inertia_tag = adom.createElement("inertia")
        inertia_tag.setAttribute("ixx", "0.03")
        inertia_tag.setAttribute("iyy", "0.03")
        inertia_tag.setAttribute("izz", "0.03")
        inertia_tag.setAttribute("ixy", "0")
        inertia_tag.setAttribute("ixz", "0")
        inertia_tag.setAttribute("iyx", "0")
        inertial_tag.appendChild(mass_tag)
        inertial_tag.appendChild(inertia_tag)
        link_tag.appendChild(vis_tag)
        link_tag.appendChild(coll_tag)
        link_tag.appendChild(inertial_tag)
        contact_tag = adom.createElement("contact")
        lateral_friction_tag = adom.createElement("lateral_friction")
        spinning_friction_tag = adom.createElement("spinning_friction")
        rolling_friction_tag = adom.createElement("rolling_friction")
        if self.name == "0":
            lateral_friction_tag.setAttribute("value", "0.1")
            spinning_friction_tag.setAttribute("value", "0.05")
            rolling_friction_tag.setAttribute("value", "0.01")
        else:
            lateral_friction_tag.setAttribute("value", "100.0")
            spinning_friction_tag.setAttribute("value", "100.0")
            rolling_friction_tag.setAttribute("value", "10.0")
        contact_tag.appendChild(lateral_friction_tag)
        contact_tag.appendChild(spinning_friction_tag)
        contact_tag.appendChild(rolling_friction_tag)
        link_tag.appendChild(contact_tag)
        return link_tag

    def to_joint_element(self, adom, parent_name=None):
        joint_tag = adom.createElement("joint")
        joint_tag.setAttribute("name", f"joint_{self.name}")
        joint_tag.setAttribute("type", "revolute")
        parent = adom.createElement("parent")
        parent.setAttribute("link", str(parent_name) if parent_name is not None else str(self.parent_name))
        joint_tag.appendChild(parent)
        child = adom.createElement("child")
        child.setAttribute("link", str(self.name))
        joint_tag.appendChild(child)
        axis = adom.createElement("axis")
        axis.setAttribute("xyz", f"{self.joint_axis_xyz:.2f} {self.joint_origin_xyz_2:.2f} {self.joint_origin_xyz_3:.2f}")
        joint_tag.appendChild(axis)
        limit = adom.createElement("limit")
        limit.setAttribute("effort", "10")
        limit.setAttribute("upper", f"{np.pi:.2f}")
        limit.setAttribute("lower", f"{-np.pi:.2f}")
        limit.setAttribute("velocity", "1")
        joint_tag.appendChild(limit)
        origin = adom.createElement("origin")
        origin.setAttribute("rpy", f"{self.joint_origin_rpy_1:.2f} {self.joint_origin_rpy_2:.2f} {self.joint_origin_rpy_3:.2f}")
        origin.setAttribute("xyz", f"{self.joint_origin_xyz_1:.2f} {self.joint_origin_xyz_2:.2f} {self.joint_origin_xyz_3:.2f}")
        joint_tag.appendChild(origin)
        return joint_tag

    def to_foot_element(self, adom):
        foot_tag = adom.createElement("link")
        foot_tag.setAttribute("name", f"foot_{self.name}")
        vis_tag = adom.createElement("visual")
        geom_tag = adom.createElement("geometry")
        sphere_tag = adom.createElement("sphere")
        sphere_tag.setAttribute("radius", str(self.link_radius * 1.2))
        geom_tag.appendChild(sphere_tag)
        vis_tag.appendChild(geom_tag)
        coll_tag = adom.createElement("collision")
        c_geom_tag = adom.createElement("geometry")
        c_sphere_tag = adom.createElement("sphere")
        c_sphere_tag.setAttribute("radius", str(self.link_radius * 1.2))
        c_geom_tag.appendChild(c_sphere_tag)
        coll_tag.appendChild(c_geom_tag)
        inertial_tag = adom.createElement("inertial")
        mass_tag = adom.createElement("mass")
        mass_tag.setAttribute("value", str(self.link_mass * 0.5))
        inertia_tag = adom.createElement("inertia")
        inertia_tag.setAttribute("ixx", "0.01")
        inertia_tag.setAttribute("iyy", "0.01")
        inertia_tag.setAttribute("izz", "0.01")
        inertia_tag.setAttribute("ixy", "0")
        inertia_tag.setAttribute("ixz", "0")
        inertia_tag.setAttribute("iyx", "0")
        inertial_tag.appendChild(mass_tag)
        inertial_tag.appendChild(inertia_tag)
        foot_tag.appendChild(vis_tag)
        foot_tag.appendChild(coll_tag)
        foot_tag.appendChild(inertial_tag)
        contact_tag = adom.createElement("contact")
        lateral_friction_tag = adom.createElement("lateral_friction")
        spinning_friction_tag = adom.createElement("spinning_friction")
        rolling_friction_tag = adom.createElement("rolling_friction")
        lateral_friction_tag.setAttribute("value", "200.0")
        spinning_friction_tag.setAttribute("value", "200.0")
        rolling_friction_tag.setAttribute("value", "20.0")
        contact_tag.appendChild(lateral_friction_tag)
        contact_tag.appendChild(spinning_friction_tag)
        contact_tag.appendChild(rolling_friction_tag)
        foot_tag.appendChild(contact_tag)
        return foot_tag
            



