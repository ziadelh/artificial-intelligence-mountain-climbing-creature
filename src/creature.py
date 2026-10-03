import genome 
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np

class MotorType(Enum):
    PULSE = 1
    SINE = 2

class Motor:
    def __init__(self, control_waveform, control_amp, control_freq, control_phase):
        if control_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        self.amp = control_amp
        self.freq = control_freq
        self.phase = control_phase
        self.step_count = 0
    def get_output(self):
        time = self.step_count / 240.0
        if self.motor_type == MotorType.SINE:
            output = self.amp * np.sin(2 * np.pi * self.freq * time + self.phase)
        else:
            output = self.amp * (1 if np.sin(2 * np.pi * self.freq * time + self.phase) > 0 else -1)
        self.step_count += 1
        return output * 3

class Creature:
    def __init__(self, gene_count, chain_init=False):
        self.spec = genome.Genome.get_gene_spec()
        self.dna = genome.Genome.get_random_genome(len(self.spec), gene_count, chain_init=chain_init)
        if isinstance(self.dna, np.ndarray):
            if len(self.dna.shape) == 2:
                self.dna = [self.dna[i, :] for i in range(self.dna.shape[0])]
            elif len(self.dna.shape) == 1:
                self.dna = [self.dna]
        if len(self.dna) < 3:
            needed = 3 - len(self.dna)
            for _ in range(needed):
                self.dna.append(genome.Genome.get_random_gene(len(self.spec), chain_init=chain_init))
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        self.get_motors()
    def get_flat_links(self):
        if self.flat_links is None:
            gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)
        return self.flat_links
    def get_expanded_links(self):
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links
        self.exp_links = self.flat_links
        return self.exp_links
    def to_xml(self, debug_print=False):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        for i, link in enumerate(self.exp_links):
            robot_tag.appendChild(link.to_link_element(adom))
            if i > 0:
                foot = link.to_foot_element(adom)
                if foot is not None:
                    robot_tag.appendChild(foot)
                    joint_tag = adom.createElement("joint")
                    joint_tag.setAttribute("name", f"joint_{link.name}_foot")
                    joint_tag.setAttribute("type", "fixed")
                    parent = adom.createElement("parent")
                    parent.setAttribute("link", link.name)
                    joint_tag.appendChild(parent)
                    child = adom.createElement("child")
                    child.setAttribute("link", f"foot_{link.name}")
                    joint_tag.appendChild(child)
                    origin = adom.createElement("origin")
                    origin.setAttribute("rpy", "0 0 0")
                    origin.setAttribute("xyz", f"0 0 {link.link_length/2}")
                    joint_tag.appendChild(origin)
                    robot_tag.appendChild(joint_tag)
        for i in range(1, len(self.exp_links)):
            parent_name = self.exp_links[i].parent_name
            robot_tag.appendChild(self.exp_links[i].to_joint_element(adom, parent_name=parent_name))
        robot_tag.setAttribute("name", "pepe")
        urdf_str = '<?xml version="1.0"?>' + robot_tag.toprettyxml()
        return urdf_str
    def get_motors(self):
        self.get_expanded_links()
        if self.motors is None:
            motors = []
            for i in range(1, len(self.exp_links)):
                l = self.exp_links[i]
                m = Motor(l.control_waveform, l.control_amp, l.control_freq, l.control_phase)
                motors.append(m)
            self.motors = motors
        return self.motors
    def update_position(self, pos):
        if self.start_position is None:
            self.start_position = pos
        else:
            self.last_position = pos
    def get_distance_travelled(self):
        if self.start_position is None or self.last_position is None:
            return 0
        p1 = np.asarray(self.start_position)
        p2 = np.asarray(self.last_position)
        return np.linalg.norm(p1 - p2)
    def update_dna(self, dna):
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        self.get_motors()
    @classmethod
    def from_genome(cls, genome_data):
        obj = cls(len(genome_data), chain_init=False)
        obj.update_dna(genome_data)
        return obj
