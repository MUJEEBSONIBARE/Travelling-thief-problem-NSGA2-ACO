# Travelling Thief Problem Solver (NSGA-II + ACO)

A Python implementation combining **Non-dominated Sorting Genetic Algorithm II (NSGA-II)** with **Ant Colony Optimization (ACO)** to solve the **Travelling Thief Problem (TTP)**. This repository includes:

- A parser for bTTP dataset files (`parse_bttp_file`).
- An ACO-based route construction module for the travelling salesman component.
- A greedy knapsack solver with randomized capacity sampling.
- NSGA-II framework to evolve and optimize time and profit objectives simultaneously.

---

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

- **bTTP Parser**: Reads `.txt` files in bTTP format and extracts metadata, node coordinates, and item lists.
- **Ant Colony Optimization**: Builds efficient TSP routes with pheromone-based learning.
- **Knapsack Solver**: Generates subset selection plans based on profit/weight ratio and randomized capacity.
- **NSGA-II Evolutionary Loop**: Combines Pareto-based selection, crowding distance, and genetic operators for multi-objective optimization.
- **Parallel Execution**: Utilizes Python's `concurrent.futures` and OpenMP-style loops (via NumPy) for speed.

---

## 🔧 Prerequisites

- **Python 3.7+**
- **NumPy**
- (Optional) A modern CPU for parallel execution

Install Python packages with:
```bash
pip install -r requirements.txt
```

**`requirements.txt`** example:
```
numpy>=1.18.0
```

---

## 🚀 Installation

1. **Clone the repository**:
   ```bash
git clone https://github.com/yourusername/travelling-thief-aco-nsga2.git
cd travelling-thief-aco-nsga2
```

2. **Create a virtual environment** (recommended):
   ```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\\Scripts\\activate  # Windows
```

3. **Install dependencies**:
   ```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. **Prepare your bTTP dataset file** (e.g., `pla33810-n33809.txt`) in the project root.

2. **Run the solver**:
   ```bash
python pla33810-n33809.py
```

3. **View results**:
   - NSGA-II outputs the final Pareto front solutions to the console.
   - A distance matrix is stored in `distances.dat` for reuse.

---

## 📂 Project Structure

```
├── pla33810-n33809.py   # Main NSGA-II + ACO implementation
├── pla33810-n33809.txt  # Sample bTTP dataset (not included)
├── distances.dat        # Memory-mapped distance matrix (generated at run-time)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation (this file)
```

---

## ⚙️ Configuration

- **Algorithm parameters** are defined at the top of `pla33810-n33809.py`:
  ```python
  V_MAX = 1.0; V_MIN = 0.1
  num_ants = 1; num_iterations = 1
  ALPHA = 1.0; BETA = 5.0
  EVAPORATION_RATE = 0.3; PHEROMONE_DEPOSIT = 100
  ```
- **NSGA-II settings** like `generations`, `population_size`, `mutation_rate`, and `crossover_rate` can be adjusted in the `nsga2_with_aco(...)` call at the bottom.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a Pull Request on GitHub.

Please adhere to PEP8 guidelines and include tests where applicable.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

