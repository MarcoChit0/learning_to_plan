# 🧠 Learning to Plan

Train LLMs for classical planning using test-time compute.

---

## 🔧 Development Commands

### 🐍 Conda

```bash
conda env create -n learning_to_plan -f environment.yml
conda activate learning_to_plan
pip install -e .
```


### 🐳 Docker

```bash
make build          # Build Docker image
make run            # Run the app
make shell          # Drop into a shell
make rebuild        # Force rebuild from scratch
make clean          # Clean up Docker cache
```

