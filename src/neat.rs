use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
};

use physics_reinforcement_learning_environment::{
    egui, Agent, Algorithm, Environment, Move, Receiver, Sender, TrainingDetails, World,
};
use rand::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct ConnectionGene {
    pub in_node: usize,
    pub out_node: usize,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: usize,
}

#[derive(Clone, Debug)]
pub struct Genome {
    pub connections: Vec<ConnectionGene>,
}

impl Genome {
    fn difference(genome1: &Genome, genome2: &Genome) -> f32 {
        let max_length = genome1.connections.len().max(genome2.connections.len());
        if max_length == 0 {
            return 0.0;
        }

        let genome1_innovations = genome1
            .connections
            .iter()
            .map(|connection| (connection.innovation, connection.weight))
            .collect::<HashMap<usize, f32>>();

        let mut common_gene_diff = 0.0;
        let mut number_of_common_genes = 0;
        for connection in &genome2.connections {
            if let Some(genome1_weight) = genome1_innovations.get(&connection.innovation) {
                common_gene_diff += 0.4 * (*genome1_weight - connection.weight).abs();
                number_of_common_genes += 1;
            }
        }
        if number_of_common_genes > 0 {
            common_gene_diff /= number_of_common_genes as f32;
        }

        let number_of_non_common_genes =
            genome1.connections.len() + genome2.connections.len() - number_of_common_genes;
        let non_common_gene_diff = number_of_non_common_genes as f32 / max_length as f32;

        common_gene_diff + non_common_gene_diff
    }
}

#[derive(Clone, Copy, Debug)]
struct NetworkConnection {
    in_node: usize,
    out_node: usize,
    weight: f32,
}

// Right now, we will have 5 input nodes (0-4), and 3 output nodes (5-7).
#[derive(Clone, Debug)]
pub struct Network {
    memory: Vec<f32>,
    normal_connections: Vec<NetworkConnection>,
    recurrent_connections: Vec<NetworkConnection>,
    genome: Genome,
    layer_index: Vec<usize>,
    topological_order: Vec<usize>,
}

impl Network {
    pub fn from_genome(genome: &Genome) -> Network {
        // Map genome nodes to a contiguous range of indices 0 to n.
        let mut genome_node_to_index = HashMap::new();
        for i in 0..8 {
            // Input and output nodes
            genome_node_to_index.insert(i, i);
        }
        for connection in &genome.connections {
            if connection.enabled {
                if !genome_node_to_index.contains_key(&connection.in_node) {
                    genome_node_to_index.insert(connection.in_node, genome_node_to_index.len());
                }
                if !genome_node_to_index.contains_key(&connection.out_node) {
                    genome_node_to_index.insert(connection.out_node, genome_node_to_index.len());
                }
            }
        }

        let mut normal_connections: Vec<NetworkConnection> = vec![];
        let mut recurrent_connections = vec![];
        for genome_connection in &genome.connections {
            if !genome_connection.enabled {
                continue;
            }

            // Topological sort
            let mut adjacency_list = vec![vec![]; genome_node_to_index.len()];
            let mut indegrees = vec![0; genome_node_to_index.len()];
            for connection in &normal_connections {
                adjacency_list[connection.in_node].push(connection.out_node);
                indegrees[connection.out_node] += 1;
            }
            adjacency_list[genome_node_to_index[&genome_connection.in_node]]
                .push(genome_node_to_index[&genome_connection.out_node]);
            indegrees[genome_node_to_index[&genome_connection.out_node]] += 1;

            let mut queue = VecDeque::new();
            for (i, indegree) in indegrees.iter().enumerate() {
                if *indegree == 0 {
                    queue.push_back(i);
                }
            }
            loop {
                let Some(node) = queue.pop_front() else { break };
                for &out_node in &adjacency_list[node] {
                    indegrees[out_node] -= 1;
                    if indegrees[out_node] == 0 {
                        queue.push_back(out_node);
                    }
                }
            }

            let connection = NetworkConnection {
                in_node: genome_node_to_index[&genome_connection.in_node],
                out_node: genome_node_to_index[&genome_connection.out_node],
                weight: genome_connection.weight,
            };
            let is_dag = indegrees.iter().all(|&indegree| indegree == 0);
            if is_dag {
                normal_connections.push(connection);
            } else {
                recurrent_connections.push(connection);
            }
        }

        // Topological sort
        let mut adjacency_list = vec![vec![]; genome_node_to_index.len()];
        let mut indegrees = vec![0; genome_node_to_index.len()];
        for connection in &normal_connections {
            adjacency_list[connection.in_node].push(connection.out_node);
            indegrees[connection.out_node] += 1;
        }

        let mut topological_order = vec![];
        let mut layers = vec![HashSet::new()];

        for (i, indegree) in indegrees.iter().enumerate() {
            if *indegree == 0 {
                layers[0].insert(i);
            }
        }

        loop {
            let mut next_layer = HashSet::new();
            for &node in layers.last().unwrap() {
                topological_order.push(node);
                for &out_node in &adjacency_list[node] {
                    indegrees[out_node] -= 1;
                    if indegrees[out_node] == 0 {
                        next_layer.insert(out_node);
                    }
                }
            }

            if next_layer.is_empty() {
                break;
            } else {
                layers.push(next_layer);
            }
        }

        // Seperate 0,1,2,3,4 and 5,6,7 into the first and last layer.
        // Remove empty layers created in the process.
        for layer in &mut layers {
            for i in 0..8 {
                layer.remove(&i);
            }
        }
        layers.insert(0, HashSet::from([0, 1, 2, 3, 4]));
        layers.push(HashSet::from([5, 6, 7]));
        layers.retain(|layer| !layer.is_empty());

        let mut topological_index = vec![0; genome_node_to_index.len()];
        for (i, &node) in topological_order.iter().enumerate() {
            topological_index[node] = i;
        }
        normal_connections.sort_by_key(|connection| topological_index[connection.out_node]);

        let mut layer_index = vec![0; genome_node_to_index.len()];
        for (i, layer) in layers.iter().enumerate() {
            for &j in layer {
                layer_index[j] = i;
            }
        }

        Network {
            memory: vec![0.0; genome_node_to_index.len()],
            normal_connections,
            recurrent_connections,
            genome: genome.clone(),
            layer_index,
            topological_order,
        }
    }
}

impl Agent for Network {
    fn details_ui(&self, ui: &mut egui::Ui, _environment: &Environment) {
        let desired_size = egui::vec2(200.0, 200.0);
        let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::click());

        if ui.is_rect_visible(rect) {
            ui.painter()
                .rect(rect, 0.0, egui::Color32::WHITE, egui::Stroke::NONE);

            let mut layers = vec![HashMap::new(); self.layer_index[5] + 1]; // self.layer_index[5] is the last layers index.
            for (i, &layer_index) in self.layer_index.iter().enumerate() {
                let index_in_layer = layers[layer_index].len();
                layers[layer_index].insert(i, index_in_layer);
            }

            for connection in &self.normal_connections {
                let in_node = connection.in_node;
                let layer_index = self.layer_index[in_node];
                let layer_size = layers[self.layer_index[in_node]].len();
                let position_in_layer = layers[self.layer_index[in_node]][&in_node];
                let x1 = 10.0 + 180.0 * (position_in_layer as f32 / (layer_size - 1) as f32);
                let y1 = 10.0 + 180.0 * (layer_index as f32 / (layers.len() - 1) as f32);

                let out_node = connection.out_node;
                let layer_index = self.layer_index[out_node];
                let layer_size = layers[self.layer_index[out_node]].len();
                let position_in_layer = layers[self.layer_index[out_node]][&out_node];
                let x2 = 10.0 + 180.0 * (position_in_layer as f32 / (layer_size - 1) as f32);
                let y2 = 10.0 + 180.0 * (layer_index as f32 / (layers.len() - 1) as f32);

                let color = if connection.weight > 0.0 {
                    egui::Color32::GREEN.gamma_multiply(0.5 + 0.5 * connection.weight.abs() / 10.0)
                } else {
                    egui::Color32::RED.gamma_multiply(0.5 + 0.5 * connection.weight.abs() / 10.0)
                };

                ui.painter().arrow(
                    rect.left_top() + egui::vec2(x1, y1),
                    egui::vec2(x2, y2) - egui::vec2(x1, y1),
                    egui::Stroke::new(3.0, color),
                );
            }

            for connection in &self.recurrent_connections {
                let in_node = connection.in_node;
                let layer_index = self.layer_index[in_node];
                let layer_size = layers[self.layer_index[in_node]].len();
                let position_in_layer = layers[self.layer_index[in_node]][&in_node];
                let x1 = 10.0 + 180.0 * (position_in_layer as f32 / (layer_size - 1) as f32);
                let y1 = 10.0 + 180.0 * (layer_index as f32 / (layers.len() - 1) as f32);

                let out_node = connection.out_node;
                let layer_index = self.layer_index[out_node];
                let layer_size = layers[self.layer_index[out_node]].len();
                let position_in_layer = layers[self.layer_index[out_node]][&out_node];
                let x2 = 10.0 + 180.0 * (position_in_layer as f32 / (layer_size - 1) as f32);
                let y2 = 10.0 + 180.0 * (layer_index as f32 / (layers.len() - 1) as f32);

                let color = if connection.weight > 0.0 {
                    egui::Color32::GREEN.gamma_multiply(0.5 + 0.5 * connection.weight.abs() / 10.0)
                } else {
                    egui::Color32::RED.gamma_multiply(0.5 + 0.5 * connection.weight.abs() / 10.0)
                };

                if in_node == out_node {
                    ui.painter().circle_stroke(
                        rect.left_top() + egui::vec2(x1 + 10.0, y1),
                        10.0,
                        egui::Stroke::new(3.0, color),
                    );
                } else {
                    ui.painter()
                        .add(egui::epaint::QuadraticBezierShape::from_points_stroke(
                            [
                                rect.left_top() + egui::vec2(x1, y1),
                                rect.left_top()
                                    + egui::vec2((x1 + x2) / 2.0 + 20.0, (y1 + y2) / 2.0 + 20.0),
                                rect.left_top() + egui::vec2(x2, y2),
                            ],
                            false,
                            color,
                            egui::Stroke::new(3.0, color),
                        ));
                }
            }

            for (layer_index, layer) in layers.iter().enumerate() {
                for position_in_layer in 0..layer.len() {
                    let x = 10.0 + 180.0 * (position_in_layer as f32 / (layer.len() - 1) as f32);
                    let y = 10.0 + 180.0 * (layer_index as f32 / (layers.len() - 1) as f32);
                    ui.painter().circle(
                        rect.left_top() + egui::vec2(x, y),
                        5.0,
                        egui::Color32::BLACK,
                        egui::Stroke::NONE,
                    );
                }
            }
        }

        ui.add_space(10.0);
        ui.label(format!("Genome: {:?}", self.genome));

        ui.add_space(10.0);
        ui.label(format!("Normal connections: {:?}", self.normal_connections));

        ui.add_space(10.0);
        ui.label(format!(
            "Recurrent connections: {:?}",
            self.recurrent_connections
        ));
    }

    fn get_move(&mut self, environment: &Environment) -> Move {
        let mut new_values = vec![0.0; self.memory.len()];

        let player_handle = environment.player_handle();
        let player = &environment.rigid_body_set()[player_handle];
        new_values[0] = player.position().translation.x;
        new_values[1] = player.position().translation.y;
        new_values[2] = player.linvel().x;
        new_values[3] = player.linvel().y;
        new_values[4] = 1.0;

        for connection in &self.recurrent_connections {
            new_values[connection.out_node] += self.memory[connection.in_node] * connection.weight;
        }

        let mut curr_topological_index = 0;
        for connection in &self.normal_connections {
            // Apply activation function (RELU) on earlier nodes
            while self.topological_order[curr_topological_index] != connection.out_node {
                if self.topological_order[curr_topological_index] > 7 {
                    new_values[self.topological_order[curr_topological_index]] =
                        new_values[self.topological_order[curr_topological_index]].max(0.0);
                }
                curr_topological_index += 1;
            }

            new_values[connection.out_node] += new_values[connection.in_node] * connection.weight;
        }
        while curr_topological_index < self.topological_order.len() {
            if self.topological_order[curr_topological_index] > 7 {
                new_values[self.topological_order[curr_topological_index]] =
                    new_values[self.topological_order[curr_topological_index]].max(0.0);
            }
            curr_topological_index += 1;
        }

        self.memory = new_values;

        Move {
            left: self.memory[5] > 0.0,
            right: self.memory[6] > 0.0,
            up: self.memory[7] > 0.0,
        }
    }
}

type SpeciesId = usize;

pub struct NeatMessage {
    population: Vec<Genome>,
    scores: Vec<f32>,
    species: Vec<(SpeciesId, Vec<usize>)>,
}

pub struct GenerationDetails {
    species: BTreeMap<SpeciesId, Vec<(Genome, f32)>>,
}

pub struct NeatTrainingDetails {
    receiver: Receiver<NeatMessage>,
    generation_details: Vec<GenerationDetails>,
    current_generation: Option<usize>,
    current_network: Option<Network>,
}

impl TrainingDetails<Network, NeatMessage> for NeatTrainingDetails {
    fn details_ui(&mut self, ui: &mut egui::Ui) -> Option<&Network> {
        self.current_network = None;

        ui.horizontal(|ui| {
            ui.label("Generation:");
            ui.add_space(10.0);

            let label = if let Some(i) = self.current_generation {
                format!("{i}")
            } else {
                "".to_string()
            };

            egui::ComboBox::from_id_source("Generation choice")
                .selected_text(label)
                .show_ui(ui, |ui| {
                    for i in 0..self.generation_details.len() {
                        ui.selectable_value(&mut self.current_generation, Some(i), format!("{i}"));
                    }
                });
        });

        if let Some(generation_index) = self.current_generation {
            for (species_id, genomes) in &self.generation_details[generation_index].species {
                ui.add_space(25.0);

                ui.label(format!("Species: {species_id}"));
                ui.add_space(10.0);
                ui.label(format!("Number of genomes: {}", genomes.len()));
                for (genome, score) in genomes {
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        ui.label(format!("Score: {score}"));
                        ui.add_space(10.0);
                        if ui.button("Visualize network").clicked() {
                            self.current_network = Some(Network::from_genome(genome));
                        }
                    });
                }
            }
        }

        self.current_network.as_ref()
    }

    fn receive_messages(&mut self) {
        for message in self.receiver.try_iter().take(1000) {
            let mut species = BTreeMap::new();
            for (species_id, genome_indices) in message.species {
                let genomes = genome_indices
                    .iter()
                    .map(|&genome_index| {
                        (
                            message.population[genome_index].clone(),
                            message.scores[genome_index],
                        )
                    })
                    .collect::<Vec<_>>();
                species.insert(species_id, genomes);
            }
            self.generation_details.push(GenerationDetails { species });
        }
    }
}

#[derive(Clone, Default)]
pub struct NeatAlgorithm {}

impl Algorithm<Network, NeatMessage, NeatTrainingDetails> for NeatAlgorithm {
    fn selection_ui(&mut self, _ui: &mut egui::Ui) {}

    fn train(&self, world: World, sender: Sender<NeatMessage>) {
        let mut thread_rng = rand::thread_rng();
        let mut innovation_counter = 0;
        let mut node_counter = 8;
        let mut species_counter = 0;

        // Vector of SpeciesId and representative genome from previous population.
        let mut species: Vec<(SpeciesId, Genome)> = vec![];

        let mut population = vec![
            Genome {
                connections: vec![]
            };
            1000
        ];

        loop {
            let mut scores = vec![];
            for genome in &population {
                let (mut environment, _) = Environment::from_world(&world);
                let mut network = Network::from_genome(genome);
                let mut min_distance = f32::INFINITY;

                for _ in 0..5000 {
                    let player_move = network.get_move(&environment);
                    environment.step(player_move);
                    min_distance = min_distance.min(environment.distance_to_goals().unwrap());
                }

                scores.push(min_distance);
            }

            let mut new_species = species
                .iter()
                .map(|(species_id, _)| (*species_id, vec![]))
                .collect::<Vec<_>>();
            for (i, genome) in population.iter().enumerate() {
                let mut assigned = false;

                for (j, species_representative) in
                    new_species
                        .iter()
                        .enumerate()
                        .map(|(i, (_, species_indices))| {
                            if i < species.len() {
                                // One of the earlier species, use representative from previous population.
                                (i, &species[i].1)
                            } else {
                                // One of the new species, use representative from current population.
                                (i, &population[species_indices[0]])
                            }
                        })
                {
                    if Genome::difference(genome, species_representative) < 3.0 {
                        new_species[j].1.push(i);
                        assigned = true;
                        break;
                    }
                }

                if !assigned {
                    species_counter += 1;
                    new_species.push((species_counter, vec![i]));
                }
            }
            new_species.retain(|species| !species.1.is_empty());

            if sender
                .send(NeatMessage {
                    population: population.clone(),
                    scores: scores.clone(),
                    species: new_species.clone(),
                })
                .is_err()
            {
                return;
            }

            let mut new_population = vec![];

            let mut population_and_scores =
                population.iter().zip(scores.iter()).collect::<Vec<_>>();
            population_and_scores.sort_by(|(_, score1), (_, score2)| {
                if score1 < score2 {
                    Ordering::Less
                } else if score1 > score2 {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });
            // Add best 250 genomes to the new population.
            // TODO: Mutate before adding to new_population.
            new_population.extend(
                population_and_scores
                    .iter()
                    .map(|(genome, _)| (*genome).clone())
                    .take(250),
            );

            let max_score = scores.iter().cloned().reduce(f32::max).unwrap();
            let average_species_fitness = new_species
                .iter()
                .map(|species| {
                    species
                        .1
                        .iter()
                        .map(|genome_index| max_score - scores[*genome_index] + 0.01)
                        .sum::<f32>()
                        / species.1.len() as f32
                })
                .collect::<Vec<_>>();
            let total_fitness = average_species_fitness.iter().sum::<f32>();
            let mut number_of_children = vec![]; // Number of children each species will have.
            for &fitness in &average_species_fitness {
                number_of_children.push((750.0 * fitness / total_fitness) as usize);
            }

            let mut add_connection_innovations = HashMap::new();
            let mut add_node_innovations = HashMap::new();

            for (i, (_, species_indices)) in new_species.iter().enumerate() {
                // Perform crossover and mutation and add to population.
                let mut species_indices = species_indices.clone();
                species_indices.sort_by(|genome_index1, genome_index2| {
                    if scores[*genome_index1] < scores[*genome_index2] {
                        Ordering::Less
                    } else if scores[*genome_index1] > scores[*genome_index2] {
                        Ordering::Greater
                    } else {
                        Ordering::Equal
                    }
                });
                species_indices.truncate(10);

                for _ in 0..number_of_children[i] {
                    let mut parents = species_indices.choose_multiple(&mut thread_rng, 2);
                    let parent_index1 = parents.next().unwrap();
                    let parent_index2 = parents.next().unwrap_or(parent_index1);

                    let (parent1, parent2) = if scores[*parent_index1] < scores[*parent_index2] {
                        (&population[*parent_index1], &population[*parent_index2])
                    } else {
                        (&population[*parent_index2], &population[*parent_index1])
                    };

                    let parent2_genomes = parent2
                        .connections
                        .iter()
                        .map(|connection| (connection.innovation, connection))
                        .collect::<HashMap<_, _>>();

                    let mut child = Genome {
                        connections: vec![],
                    };

                    for connection1 in &parent1.connections {
                        if let Some(connection2) = parent2_genomes.get(&connection1.innovation) {
                            let enabled = connection1.enabled && connection2.enabled
                                || random::<f32>() < 0.25;
                            if random() {
                                child.connections.push(ConnectionGene {
                                    in_node: connection1.in_node,
                                    out_node: connection1.out_node,
                                    weight: connection1.weight,
                                    enabled,
                                    innovation: connection1.innovation,
                                });
                            } else {
                                child.connections.push(ConnectionGene {
                                    in_node: connection2.in_node,
                                    out_node: connection2.out_node,
                                    weight: connection2.weight,
                                    enabled,
                                    innovation: connection2.innovation,
                                });
                            }
                        } else {
                            let enabled = connection1.enabled || random::<f32>() < 0.25;
                            child.connections.push(ConnectionGene {
                                in_node: connection1.in_node,
                                out_node: connection1.out_node,
                                weight: connection1.weight,
                                enabled,
                                innovation: connection1.innovation,
                            });
                        }
                    }

                    // Connection weight mutation
                    if random::<f32>() < 0.8 {
                        if random::<f32>() < 0.9 {
                            for connection in child.connections.iter_mut() {
                                connection.weight += 2.0 * random::<f32>() - 1.0;
                            }
                        } else {
                            for connection in child.connections.iter_mut() {
                                connection.weight = 20.0 * random::<f32>() - 10.0;
                            }
                        }
                    }

                    // Add node mutation
                    if random::<f32>() < 0.03 {
                        let mut enabled_connections = child
                            .connections
                            .iter_mut()
                            .filter(|connection| connection.enabled)
                            .collect::<Vec<_>>();
                        let connection = enabled_connections.choose_mut(&mut thread_rng);
                        if let Some(connection) = connection {
                            let connection_key = (connection.in_node, connection.out_node);
                            let (innovation, node) = if let Some((innovation, node)) =
                                add_node_innovations.get(&connection_key)
                            {
                                (*innovation, *node)
                            } else {
                                add_node_innovations
                                    .insert(connection_key, (innovation_counter, node_counter));
                                innovation_counter += 2;
                                node_counter += 1;
                                (innovation_counter - 2, node_counter - 1)
                            };
                            connection.enabled = false;

                            let connection1 = ConnectionGene {
                                in_node: connection.in_node,
                                out_node: node,
                                weight: 1.0,
                                enabled: true,
                                innovation,
                            };
                            let connection2 = ConnectionGene {
                                in_node: node,
                                out_node: connection.out_node,
                                weight: connection.weight,
                                enabled: true,
                                innovation: innovation + 1,
                            };
                            child.connections.push(connection1);
                            child.connections.push(connection2);
                        }
                    }

                    // Add connection mutation
                    if random::<f32>() < 0.05 {
                        let connections = child
                            .connections
                            .iter()
                            .map(|connection| (connection.in_node, connection.out_node))
                            .collect::<HashSet<_>>();
                        let nodes = child
                            .connections
                            .iter()
                            .flat_map(|connection| [connection.in_node, connection.out_node])
                            .chain(0..8)
                            .collect::<HashSet<_>>();
                        for _ in 0..50 {
                            let in_node = *nodes.iter().choose(&mut thread_rng).unwrap();
                            let out_node = *nodes.iter().choose(&mut thread_rng).unwrap();
                            if out_node > 4 && !connections.contains(&(in_node, out_node)) {
                                // Input nodes can't be out_node
                                let innovation = if let Some(innovation) =
                                    add_connection_innovations.get(&(in_node, out_node))
                                {
                                    *innovation
                                } else {
                                    add_connection_innovations
                                        .insert((in_node, out_node), innovation_counter);
                                    innovation_counter += 1;
                                    innovation_counter - 1
                                };

                                child.connections.push(ConnectionGene {
                                    in_node,
                                    out_node,
                                    weight: 20.0 * random::<f32>() - 10.0,
                                    enabled: true,
                                    innovation,
                                });
                                break;
                            }
                        }
                    }

                    new_population.push(child);
                }
            }

            species = new_species
                .into_iter()
                .map(|(species_id, species_indices)| {
                    (species_id, population[species_indices[0]].clone())
                })
                .collect();
            population = new_population;
        }
    }

    fn training_details_receiver(
        &self,
        _world: &World,
        receiver: Receiver<NeatMessage>,
    ) -> NeatTrainingDetails {
        NeatTrainingDetails {
            receiver,
            generation_details: vec![],
            current_generation: None,
            current_network: None,
        }
    }
}
