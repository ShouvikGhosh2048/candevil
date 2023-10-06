use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
};

use physics_reinforcement_learning_environment::{
    egui, Agent, Algorithm, Environment, Move, Receiver, Sender, TrainingDetails, World,
    WorldObject,
};

// The world will be divided into EGUI_CELL_SIZE by EGUI_CELL_SIZE cells
// and then each cell is shown by egui.
const EGUI_CELL_SIZE: f32 = 10.0;

#[derive(Clone)]
pub struct ListOfMovesAgent {
    move_repeat_count: usize,
    moves: Vec<Move>,
    current_move: usize,
    current_repeat_count: usize,
}

impl Agent for ListOfMovesAgent {
    fn get_move(&mut self, _environment: &Environment) -> Move {
        if self.current_move >= self.moves.len() {
            Move {
                left: false,
                right: false,
                up: false,
            }
        } else {
            let result_move = self.moves[self.current_move];

            self.current_repeat_count += 1;
            if self.current_repeat_count == self.move_repeat_count {
                self.current_move += 1;
                self.current_repeat_count = 0;
            }

            result_move
        }
    }

    fn details_ui(&self, ui: &mut egui::Ui, _environment: &Environment) {
        let player_move = if self.current_move < self.moves.len() {
            self.moves[self.current_move]
        } else {
            Move {
                left: false,
                right: false,
                up: false,
            }
        };
        ui.label(
            egui::RichText::new(format!("Move: {}", move_to_string(player_move)))
                .strong()
                .monospace(),
        );

        for (index, player_move) in self.moves.iter().enumerate() {
            let label = format!("{}. {}", index + 1, move_to_string(*player_move));

            ui.add_space(5.0);
            if index == self.current_move {
                ui.label(egui::RichText::new(label).strong().monospace());
            } else {
                ui.label(egui::RichText::new(label).monospace());
            }
        }
    }
}

fn move_to_string(player_move: Move) -> String {
    let mut result = "".to_string();

    result += if player_move.left { "LEFT " } else { "     " };
    result += if player_move.right {
        "RIGHT "
    } else {
        "      "
    };
    result += if player_move.up { "UP" } else { "  " };

    result
}

// While perfoming brute force search we explore the tree of possible moves.
// The order in which the node is sent from the training thread is it's id (starting from 0).
#[derive(Clone)]
pub struct BruteForceSearchNode {
    parent: Option<usize>,
    player_move: Move,
    score: f32,
    player_displacement: (f32, f32),
    player_velocity: (f32, f32),
}

pub struct BruteForceSearchTrainingDetails {
    move_repeat_count: usize,
    nodes: Vec<BruteForceSearchNode>,
    node_grid: HashMap<(i32, i32), Vec<usize>>, // HashMap from a cell to the vector of the indices of the nodes lying in the cell.
    receiver: Receiver<BruteForceSearchNode>,
    world: World,
    selected_cell: Option<(i32, i32)>,
    egui_player_offset_from_center: egui::Vec2,
    zoom: f32,
    current_agent: Option<ListOfMovesAgent>,
}

impl TrainingDetails<ListOfMovesAgent, BruteForceSearchNode> for BruteForceSearchTrainingDetails {
    fn receive_messages(&mut self) {
        for node in self.receiver.try_iter().take(1000) {
            let cell = (
                (node.player_displacement.0 / EGUI_CELL_SIZE).floor() as i32,
                (node.player_displacement.1 / EGUI_CELL_SIZE).floor() as i32,
            );
            let node_index = self.nodes.len();

            self.nodes.push(node);
            self.node_grid.entry(cell).or_default().push(node_index);
        }
    }

    fn details_ui(&mut self, ui: &mut egui::Ui) -> Option<&ListOfMovesAgent> {
        self.current_agent = None;

        ui.set_min_height(500.0);
        ui.set_min_width(900.0);

        ui.horizontal(|ui| {
            // Left panel
            ui.vertical(|ui| {
                ui.set_min_width(200.0);
                ui.set_min_height(500.0);

                egui::ScrollArea::vertical()
                    .id_source("Left detail panel")
                    .show(ui, |ui| {
                        ui.label(format!("Total number of agents: {}", self.nodes.len()));

                        ui.add_space(10.0);

                        if let Some(selected_cell) = self.selected_cell {
                            ui.label(format!(
                                "Cell: {:?}",
                                (
                                    selected_cell.0 as f32 * EGUI_CELL_SIZE,
                                    selected_cell.1 as f32 * EGUI_CELL_SIZE
                                )
                            ));

                            ui.add_space(10.0);

                            ui.label(format!(
                                "Number of agents: {}",
                                self.node_grid[&selected_cell].len()
                            ));

                            for index in &self.node_grid[&selected_cell] {
                                let node = &self.nodes[*index];

                                ui.add_space(10.0);

                                ui.label(format!("Score: {}", node.score));
                                ui.label(format!(
                                    "Final displacement: {:?}",
                                    node.player_displacement
                                ));
                                ui.label(format!("Final velocity: {:?}", node.player_velocity));

                                if ui.button("Visualise agent").clicked() {
                                    // Walk up the tree and find the list of moves.
                                    let mut moves = vec![node.player_move];
                                    let mut curr = node.parent;
                                    while let Some(index) = curr {
                                        moves.push(self.nodes[index].player_move);
                                        curr = self.nodes[index].parent;
                                    }
                                    moves.reverse();

                                    self.current_agent = Some(ListOfMovesAgent {
                                        move_repeat_count: self.move_repeat_count,
                                        moves,
                                        current_move: 0,
                                        current_repeat_count: 0,
                                    });
                                }
                            }
                        }
                    });
            });

            ui.separator();

            // Grid
            ui.vertical(|ui| {
                let desired_size = egui::vec2(700.0, 500.0);
                let (rect, mut response) =
                    ui.allocate_exact_size(desired_size, egui::Sense::click_and_drag());

                if response.clicked() {
                    response.mark_changed();

                    let mouse_offset = response.interact_pointer_pos().unwrap()
                        - rect.center()
                        - self.egui_player_offset_from_center;
                    let selected_cell = (
                        (mouse_offset.x / (10.0 * self.zoom)).floor() as i32,
                        -(mouse_offset.y / (10.0 * self.zoom)).ceil() as i32,
                    );

                    if self.node_grid.contains_key(&selected_cell) {
                        self.selected_cell = Some(selected_cell);
                    } else {
                        self.selected_cell = None;
                    }
                } else if response.dragged() {
                    self.egui_player_offset_from_center += response.drag_delta();
                } else if response.hovered() {
                    if let Some(mouse_position) = response.hover_pos() {
                        let mouse_offset = mouse_position - rect.center();
                        let zoom_delta = ui.input(|i| i.zoom_delta());
                        if !(0.99..=1.01).contains(&zoom_delta) {
                            let prev_zoom = self.zoom;
                            self.zoom *= zoom_delta;
                            self.egui_player_offset_from_center = mouse_offset
                                - self.zoom / prev_zoom
                                    * (mouse_offset - self.egui_player_offset_from_center);
                        }
                    }
                }

                // Player position in world.
                // We flip the y coordinate as Bevy and egui have different +y-axis directions.
                let world_player_position = egui::vec2(
                    self.world.player_position[0],
                    -self.world.player_position[1],
                );

                if ui.is_rect_visible(rect) {
                    let visuals = ui.style().interact_selectable(&response, false);
                    let rect = rect.expand(visuals.expansion);
                    ui.painter()
                        .rect(rect, 0.0, egui::Color32::WHITE, egui::Stroke::NONE);

                    for object in &self.world.objects {
                        // Positions in world without rotation.
                        // We flip the y coordinate as Bevy and egui have different +y-axis directions.
                        let mut points = vec![
                            egui::vec2(
                                object.position[0] - object.scale[0].abs() / 2.0,
                                -(object.position[1] - object.scale[1].abs() / 2.0),
                            ),
                            egui::vec2(
                                object.position[0] - object.scale[0].abs() / 2.0,
                                -(object.position[1] + object.scale[1].abs() / 2.0),
                            ),
                            egui::vec2(
                                object.position[0] + object.scale[0].abs() / 2.0,
                                -(object.position[1] + object.scale[1].abs() / 2.0),
                            ),
                            egui::vec2(
                                object.position[0] + object.scale[0].abs() / 2.0,
                                -(object.position[1] - object.scale[1].abs() / 2.0),
                            ),
                        ];

                        for point in &mut points {
                            // Apply rotation
                            let center = egui::vec2(object.position[0], -object.position[1]);
                            let center_offset = *point - center;
                            *point = center
                                + egui::vec2(
                                    center_offset.x * object.rotation.cos()
                                        + center_offset.y * object.rotation.sin(),
                                    center_offset.y * object.rotation.cos()
                                        - center_offset.x * object.rotation.sin(),
                                );

                            // Calculate displacement from player center.
                            *point -= world_player_position;

                            *point *= self.zoom;
                        }

                        let points = points
                            .iter()
                            .map(|point| {
                                (rect.center().to_vec2()
                                    + self.egui_player_offset_from_center
                                    + *point)
                                    .to_pos2()
                            })
                            .collect();

                        let color = match object.object {
                            WorldObject::Block { .. } => egui::Color32::BLACK,
                            WorldObject::Goal => egui::Color32::GREEN,
                        };

                        ui.painter()
                            .with_clip_rect(rect)
                            .add(egui::Shape::convex_polygon(
                                points,
                                color,
                                egui::Stroke::NONE,
                            ));
                    }

                    let max_indices_count = self
                        .node_grid
                        .values()
                        .map(|indices| indices.len())
                        .max()
                        .unwrap_or(0);

                    for (cell, indices) in self.node_grid.iter() {
                        let min = rect.center()
                            + self.egui_player_offset_from_center
                            + egui::vec2(
                                (cell.0 as f32) * 10.0 * self.zoom,
                                -(cell.1 as f32 + 1.0) * 10.0 * self.zoom,
                            );
                        let max = min + egui::vec2(10.0 * self.zoom, 10.0 * self.zoom);
                        let fraction = indices.len() as f32 / max_indices_count as f32;
                        let color = (125.0 + 75.0 * (1.0 - fraction)) as u8;

                        ui.painter().with_clip_rect(rect).rect(
                            egui::Rect { min, max },
                            0.0,
                            egui::Color32::from_gray(color),
                            egui::Stroke::NONE,
                        );
                    }

                    if let Some(selected_cell) = self.selected_cell {
                        let min = rect.center()
                            + self.egui_player_offset_from_center
                            + egui::vec2(
                                (selected_cell.0 as f32) * 10.0 * self.zoom,
                                -(selected_cell.1 as f32 + 1.0) * 10.0 * self.zoom,
                            );
                        let max = min + egui::vec2(10.0 * self.zoom, 10.0 * self.zoom);

                        ui.painter().with_clip_rect(rect).rect(
                            egui::Rect { min, max },
                            0.0,
                            egui::Color32::YELLOW,
                            egui::Stroke::NONE,
                        );
                    }
                }
            });
        });

        self.current_agent.as_ref()
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Binning {
    None,
    PositionAndVelocity,
    Position,
    PositionThenVelocity,
}

impl Display for Binning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Binning::None => write!(f, "None"),
            Binning::PositionAndVelocity => write!(f, "Position & velocity"),
            Binning::Position => write!(f, "Position"),
            Binning::PositionThenVelocity => write!(f, "Position then velocity"),
        }
    }
}

#[derive(Clone)]
pub struct BruteForceSearch {
    move_repeat_count: usize,
    binning: Binning,
    position_cell_multiplier: f32,
    velocity_cell_multiplier: f32,
    check_world_bounds: bool,
}

impl Default for BruteForceSearch {
    fn default() -> Self {
        BruteForceSearch {
            move_repeat_count: 50,
            binning: Binning::PositionAndVelocity,
            position_cell_multiplier: 0.2,
            velocity_cell_multiplier: 1.0,
            check_world_bounds: true,
        }
    }
}

#[derive(PartialEq, Eq, Hash)]
struct PositionCell(i32, i32);

#[derive(PartialEq, Eq, Hash)]
struct VelocityCell(i32, i32);

impl Algorithm<ListOfMovesAgent, BruteForceSearchNode, BruteForceSearchTrainingDetails>
    for BruteForceSearch
{
    fn training_details_receiver(
        &self,
        world: &World,
        receiver: Receiver<BruteForceSearchNode>,
    ) -> BruteForceSearchTrainingDetails {
        BruteForceSearchTrainingDetails {
            move_repeat_count: self.move_repeat_count,
            nodes: vec![],
            node_grid: HashMap::new(),
            receiver,
            world: world.clone(),
            selected_cell: None,
            egui_player_offset_from_center: egui::vec2(0.0, 0.0),
            zoom: 1.0,
            current_agent: None,
        }
    }

    fn selection_ui(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Brute Force Search").heading());

        egui::Grid::new("Selection UI")
            .spacing([10.0, 10.0])
            .show(ui, |ui| {
                ui.label("Move repeat count:");
                ui.add(egui::DragValue::new(&mut self.move_repeat_count).clamp_range(1..=100));
                ui.end_row();

                ui.label("Binning");
                egui::ComboBox::from_id_source("Binning")
                    .selected_text(format!("{}", &self.binning))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.binning, Binning::None, "None");
                        ui.selectable_value(
                            &mut self.binning,
                            Binning::PositionAndVelocity,
                            "Position & Velocity",
                        );
                        ui.selectable_value(&mut self.binning, Binning::Position, "Position");
                        ui.selectable_value(
                            &mut self.binning,
                            Binning::PositionThenVelocity,
                            "Position then Velocity",
                        );
                    });
                ui.end_row();

                ui.label("Position cell multiplier:");
                ui.add(
                    egui::DragValue::new(&mut self.position_cell_multiplier)
                        .clamp_range(0.001..=100.0),
                );
                ui.end_row();

                ui.label("Velocity cell multiplier:");
                ui.add(
                    egui::DragValue::new(&mut self.velocity_cell_multiplier)
                        .clamp_range(0.001..=100.0),
                );
                ui.end_row();

                ui.label("Check world bounds:");
                ui.checkbox(&mut self.check_world_bounds, "");
                ui.end_row();
            });
    }

    fn train(&self, world: World, sender: Sender<BruteForceSearchNode>) {
        // Find world bounds
        let mut world_x_min = f32::INFINITY;
        let mut world_x_max = f32::NEG_INFINITY;
        let mut world_y_min = f32::INFINITY;
        let mut world_y_max = f32::NEG_INFINITY;

        for object in &world.objects {
            for x_dir in [-1.0, 1.0] {
                for y_dir in [-1.0, 1.0] {
                    let point = (
                        object.position[0]
                            + object.rotation.cos() * x_dir * object.scale[0].abs() / 2.0
                            - object.rotation.sin() * y_dir * object.scale[1].abs() / 2.0,
                        object.position[1]
                            + object.rotation.sin() * x_dir * object.scale[0].abs() / 2.0
                            + object.rotation.cos() * y_dir * object.scale[1].abs() / 2.0,
                    );

                    world_x_min = world_x_min.min(point.0);
                    world_x_max = world_x_max.max(point.0);
                    world_y_min = world_y_min.min(point.1);
                    world_y_max = world_y_max.max(point.1);
                }
            }
        }

        let mut nodes = vec![];

        let mut queue1 = VecDeque::new();
        let mut queue2 = VecDeque::new();

        let mut visited_states: HashMap<PositionCell, HashSet<VelocityCell>> = HashMap::new();

        let mut visited_root_children = false;

        loop {
            let parent = if !visited_root_children {
                // First iteration of the loop visits the children of the root.
                visited_root_children = true;
                None
            } else if let Some(index) = queue1.pop_front() {
                Some(index)
            } else if let Some(index) = queue2.pop_front() {
                Some(index)
            } else {
                break;
            };

            for left in [false, true] {
                for right in [false, true] {
                    for up in [false, true] {
                        if !left || !right {
                            let player_move = Move { left, right, up };
                            // Collect all the moves by walking up the tree.
                            let mut parent_moves = vec![];
                            let mut curr_node = parent;
                            while let Some(index) = curr_node {
                                let (curr_parent, curr_move) = nodes[index];
                                parent_moves.push(curr_move);
                                curr_node = curr_parent;
                            }
                            parent_moves.reverse();

                            let (mut environment, _) = Environment::from_world(&world);
                            let mut score = f32::INFINITY;

                            for parent_move in parent_moves {
                                for _ in 0..self.move_repeat_count {
                                    environment.step(parent_move);
                                    score = score.min(environment.distance_to_goals().unwrap());
                                }
                            }

                            for _ in 0..self.move_repeat_count {
                                environment.step(player_move);
                                score = score.min(environment.distance_to_goals().unwrap());
                            }

                            let player_handle = environment.player_handle();
                            let player = &environment.rigid_body_set()[player_handle];

                            // We divide by 0.00625 to convert from environment scale to world scale.
                            let player_displacement = (
                                player.position().translation.x / 0.00625
                                    - world.player_position[0],
                                player.position().translation.y / 0.00625
                                    - world.player_position[1],
                            );
                            let player_velocity =
                                (player.linvel().x / 0.00625, player.linvel().y / 0.00625);

                            let position_cell = PositionCell(
                                (player_displacement.0
                                    / (self.position_cell_multiplier
                                        * self.move_repeat_count as f32))
                                    .floor() as i32,
                                (player_displacement.1
                                    / (self.position_cell_multiplier
                                        * self.move_repeat_count as f32))
                                    .floor() as i32,
                            );
                            let velocity_cell = VelocityCell(
                                (player_velocity.0
                                    / (self.velocity_cell_multiplier
                                        * self.move_repeat_count as f32))
                                    .floor() as i32,
                                (player_velocity.1
                                    / (self.velocity_cell_multiplier
                                        * self.move_repeat_count as f32))
                                    .floor() as i32,
                            );

                            let in_bounding_box = world_x_min - 50.0
                                < world.player_position[0] + player_displacement.0
                                && world.player_position[0] + player_displacement.0
                                    < world_x_max + 50.0
                                && world_y_min - 50.0
                                    < world.player_position[1] + player_displacement.1
                                && world.player_position[1] + player_displacement.1
                                    < world_x_max + 250.0;

                            if !self.check_world_bounds || in_bounding_box {
                                match &self.binning {
                                    Binning::None => {
                                        // We will always explore the children of the new node.
                                        queue1.push_back(nodes.len());
                                    }
                                    Binning::PositionAndVelocity => {
                                        // We will explore the children of the new node
                                        // if it's position or velocity lie in a different bin.
                                        if let Some(velocities) =
                                            visited_states.get_mut(&position_cell)
                                        {
                                            if !velocities.contains(&velocity_cell) {
                                                velocities.insert(velocity_cell);
                                                queue1.push_back(nodes.len());
                                            }
                                        } else {
                                            let mut velocities = HashSet::new();
                                            velocities.insert(velocity_cell);
                                            visited_states.insert(position_cell, velocities);

                                            queue1.push_back(nodes.len());
                                        }
                                    }
                                    Binning::Position => {
                                        // We will explore the children of the new node
                                        // if it's position lies in a different bin.
                                        if visited_states.get_mut(&position_cell).is_none() {
                                            let mut velocities = HashSet::new();
                                            velocities.insert(velocity_cell);
                                            visited_states.insert(position_cell, velocities);

                                            queue1.push_back(nodes.len());
                                        }
                                    }
                                    Binning::PositionThenVelocity => {
                                        // We will explore the children of the new node
                                        // if it's position or velocity lies in a different bin.
                                        // However, nodes in a new position bin have a higher priority.
                                        if let Some(velocities) =
                                            visited_states.get_mut(&position_cell)
                                        {
                                            if !velocities.contains(&velocity_cell) {
                                                velocities.insert(velocity_cell);
                                                queue2.push_back(nodes.len());
                                            }
                                        } else {
                                            let mut velocities = HashSet::new();
                                            velocities.insert(velocity_cell);
                                            visited_states.insert(position_cell, velocities);

                                            queue1.push_back(nodes.len());
                                        }
                                    }
                                }
                            }

                            nodes.push((parent, player_move));

                            if sender
                                .send(BruteForceSearchNode {
                                    parent,
                                    player_move,
                                    score,
                                    player_displacement,
                                    player_velocity,
                                })
                                .is_err()
                                || score < 1e-10
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}
