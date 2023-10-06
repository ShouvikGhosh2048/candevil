use physics_reinforcement_learning_environment::run;

mod brute_force_search;
mod neat;

use neat::{NeatAlgorithm, NeatMessage, NeatTrainingDetails, Network};

fn main() {
    run::<Network, NeatMessage, NeatTrainingDetails, NeatAlgorithm>();
}
