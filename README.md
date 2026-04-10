# GLOFAIR

`GLOFAIR` is a methodology for learning **fair** and **high-performing** neural networks in Federated Learning cross-silo settings. By combining predictive performance and fairness requirements into a single objective,`GLOFAIR`  integrates fairness into the training process, defining a specific scoring function and a corresponding loss function. This approach enables the learning of models that achieve a **balanced trade-off** between fairness and predictive performance, without requiring the computation of the Pareto Front. Unlike many existing methods that focus on a single fairness metric or sensitive attribute, `GLOFAIR` supports **multiple constraints** simultaneously. These constraints can involve different fairness metrics, such as Demographic Parity and Equalized Odds, and can be defined on both binary and non-binary sensitive attributes.

---

## **Installation**

We recommend setting up a new Conda environment with **Python >= 3.9**.

### **1. Create a Conda environment**
```bash
conda create -n "glofair" python==3.9
```

### **2. Activate the environment**
```bash
conda activate glofair
```

### **3. Clone the repository**
```bash
git clone https://github.com/michelefontana92/GLOFAIR
```

### **4. Navigate to the project directory**
```bash
cd GLOFAIR
```

### **5. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## **Project Structure**
The project follows this structure:

```bash
.
├── LICENSE
├── README.md
├── data
│   ├── Adult
│   ├── Compas
│   └── Credit
├── requirements.txt
└── src
    ├── architectures
    ├── builder
    ├── callbacks
    ├── client
    ├── dataloaders
    ├── loggers
    ├── main.py
    ├── metrics
    ├── requirements
    ├── runs
    ├── server
    ├── surrogates
    └── wrappers
```

- **`data/`** → Contains federated datasets used in experiments.
- **`src/`** → Contains the core implementation of GLOFAIR.

---

## **Usage**
To run an experiment with `GLOFAIR`:

### **1. Navigate to the `src` directory**
```bash
cd src
```

### **2. Execute `main.py` with options**
```bash
python main.py --options
```

### **Available Options**
```bash
Options:
  -r, --run TEXT              Name of the run to execute
  -p, --project_name TEXT     Name of the WandB project
  -n, --num_clients INTEGER   Number of clients (default = 10)
  -ml, --metrics_list TEXT    List of fairness metrics
  -gl, --groups_list TEXT     List of sensitive groups
  -tl, --threshold_list FLOAT Threshold values for fairness constraints
  -g, --gpu_devices TEXT      List of GPUs to use
```

#### **Predefined Runs (`runs` folder)**
- **`adult_glofair`** → Uses the **Adult** dataset.
- **`compas_glofair`** → Uses the **Compas** dataset.
- **`credit_glofair`** → Uses the **Credit** dataset.

The code supports the following **fairness metrics**:
- `demographic_parity`
- `equal_opportunity`
- `predictive_equality`
- `equalized_odds`

---

## **Logging**
The code uses **Weights & Biases (WandB)** for experiment tracking. 
To use it:
1. Create a free account at [WandB](https://wandb.ai/site/).
2. Follow the login instructions after executing the code.

We plan to allow users to choose other logging systems in the future.

---

## **GPU Execution**
The code can be executed on a **CPU** or **GPU(s)**.

- To specify GPU(s), use the `-g` option:
    ```bash
    python main.py -g 0
    ```
- If `-g` is not provided, execution will default to CPU.

---

## **Examples**

### **1. One Fairness Constraint**
Create a federation with 10 clients enforcing **Demographic Parity (DP ≤ 0.20) on Gender** using the **Adult** dataset, running on **GPU 0**.
```bash
python main.py -r adult_glofair -ml demographic_parity -tl 0.20 -gl Gender -p Adult_Gender -g 0
```

### **2. Mixed Fairness Metrics**
Create a federation with 10 clients enforcing **DP ≤ 0.20 on Gender** and **Equalized Odds (EOD ≤ 0.20) on Gender**, using the **Adult** dataset and running on **GPU 0**.
```bash
python main.py -r adult_glofair -ml demographic_parity -tl 0.20 -gl Gender -ml equalized_odds -tl 0.20 -gl Gender -p Adult_Mixed -g 0
```

### **3. Three Fairness Constraints**
Create a federation with 10 clients enforcing **DP ≤ 0.20 on Gender, DP ≤ 0.20 on Race, and DP ≤ 0.20 on GenderRace**, using the **Adult** dataset on **GPU 0**.
```bash
python main.py -r adult_glofair -ml demographic_parity -tl 0.20 -gl Gender -ml demographic_parity -tl 0.20 -gl Race -ml demographic_parity -tl 0.20 -gl GenderRace -p Adult_ThreeConstraints -g 0
```

---

## **License**
This project is licensed under the **Apache 2.0 License**. See the [LICENSE](LICENSE) file for details.

---

## **Contact**
For any questions or suggestions, feel free to reach out:

📧 **Email**: [michele.fontana@phd.unipi.it](mailto:michele.fontana@phd.unipi.it)
