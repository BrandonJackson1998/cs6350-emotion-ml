# Contributing Guide 🤝

Thank you for your interest in contributing to the cs6350-emotion-ml project! This guide will help you get started.

## Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest new features or improvements
- 📝 Improve documentation
- 🔧 Submit code changes
- 🧪 Add tests
- 🎨 Improve visualizations

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cs6350-emotion-ml.git
   cd cs6350-emotion-ml
   ```
3. **Set up the development environment** (see [README](README.md))
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### 1. Make Your Changes

- Keep changes focused and atomic
- Follow the existing code style
- Add comments where necessary
- Update documentation if needed

### 2. Test Your Changes

```bash
# Activate virtual environment
source .virtual_environment/bin/activate

# Run a quick test
python -m src.benchmark  # or your modified code

# Verify imports work
python -c "from src.benchmark import create_experiment_config; print('✓ Import successful')"
```

### 3. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description of change"
```

**Good commit messages:**
- ✅ "Add sampling weight parameter to experiment config"
- ✅ "Fix: Handle missing image files gracefully"
- ✅ "Docs: Add troubleshooting section for GPU issues"

**Bad commit messages:**
- ❌ "Update code"
- ❌ "Fix bug"
- ❌ "Changes"

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Why the change is needed
- Any relevant issue numbers
- Screenshots (for UI changes)

## Code Style Guidelines

### Python Code

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small

**Example:**
```python
def create_experiment_config(experiment_name, sampling_weights=None, **kwargs):
    """
    Create experiment configuration with specified parameters.
    
    Args:
        experiment_name (str): Name of the experiment
        sampling_weights (dict, optional): Weights for each emotion class
        **kwargs: Additional configuration parameters
        
    Returns:
        dict: Configuration dictionary with all parameters
    """
    config = {
        'experiment_name': experiment_name,
        'sampling_weights': sampling_weights,
        # ... more config
    }
    return config
```

### Documentation

- Use clear, simple language
- Include code examples
- Add emojis for better readability (sparingly)
- Update README if adding new features

## Types of Contributions

### Bug Reports

When reporting bugs, include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version)
- Error messages and stack traces

### Feature Requests

When requesting features, include:
- Clear description of the feature
- Use case and benefits
- Possible implementation approach
- Any relevant examples

### Documentation Improvements

- Fix typos and grammar
- Clarify confusing sections
- Add missing information
- Improve examples

### Code Contributions

Good areas for contribution:
- Add new emotion detection models
- Improve data augmentation
- Add visualization options
- Optimize training performance
- Add support for video input
- Improve error handling

## Testing

Before submitting:

1. **Test your changes work**:
   ```bash
   source .virtual_environment/bin/activate
   python run_custom_experiment.py  # or your test
   ```

2. **Verify no imports are broken**:
   ```bash
   python -c "from src.benchmark import *"
   ```

3. **Check for obvious errors**:
   ```bash
   python -m py_compile src/benchmark.py
   ```

## Project Structure

Understanding the structure helps with contributions:

```
src/
├── benchmark.py         # Main training/evaluation logic
                         # Good place for model improvements

run_custom_experiment.py # Example experiments
                         # Good place for experiment templates

test_fear_sadness_experiment.py  # Specific test cases
                                 # Good place for new experiments

README.md                # Main documentation
                        # Update when adding features

requirements.txt         # Dependencies
                        # Update if adding new packages
```

## Communication

- Be respectful and constructive
- Ask questions if unclear
- Provide context in discussions
- Be patient with review process

## Code Review Process

1. Maintainers will review your PR
2. They may request changes or ask questions
3. Make requested changes
4. Once approved, your PR will be merged

## Recognition

Contributors will be acknowledged in:
- GitHub contributors list
- Release notes (for significant contributions)
- Project documentation

## Questions?

- Open an issue for questions
- Reach out to maintainers
- Check existing issues and PRs

---

Thank you for contributing! 🎉
