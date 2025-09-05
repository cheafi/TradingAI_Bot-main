# Contributing to TradingAI Bot 🤝

Thank you for your interest in contributing to TradingAI Bot! We welcome contributions from developers of all skill levels.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of trading concepts (helpful but not required)
- Familiarity with Python, pandas, and machine learning (for ML contributions)

### First Time Contributors

Looking for a good first issue? Check out issues labeled with:
- `good first issue` - Simple issues perfect for newcomers
- `help wanted` - Issues where we'd appreciate community help
- `documentation` - Help improve our docs

## 🛠️ Development Setup

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/TradingAI_Bot-main.git
   cd TradingAI_Bot-main
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

3. **Configure pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Verify setup**
   ```bash
   # Run tests to ensure everything works
   pytest tests/
   
   # Run basic functionality check
   python -c "import src; print('Setup successful!')"
   ```

## 🔄 Contributing Process

### 1. Create an Issue

Before starting work, create an issue to:
- Describe the bug you want to fix
- Propose the feature you want to add
- Discuss the approach for large changes

### 2. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Changes

- Write clean, readable code
- Follow our [code standards](#code-standards)
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

```bash
# Run full test suite
pytest tests/ -v

# Run specific tests
pytest tests/test_your_component.py

# Check code formatting
black --check .
isort --check-only .

# Type checking
mypy src/
```

### 5. Commit and Push

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "feat: add new trading strategy component"

# Push to your fork
git push origin feature/your-feature-name
```

### 6. Submit Pull Request

- Go to the original repository on GitHub
- Click "New Pull Request"
- Fill out the PR template completely
- Link any related issues
- Request review from maintainers

## 📝 Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Use Black for formatting (line length: 79)
black --line-length 79 .

# Sort imports with isort
isort .

# Type hints are encouraged
def calculate_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change()
```

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **flake8**: Linting
- **pytest**: Testing

### Naming Conventions

```python
# Variables and functions: snake_case
trading_strategy = Strategy()
def calculate_sharpe_ratio():
    pass

# Classes: PascalCase
class TradingAgent:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_POSITION_SIZE = 0.1
```

### Documentation Strings

Use NumPy-style docstrings:

```python
def backtest_strategy(strategy, data, initial_capital=10000):
    """
    Backtest a trading strategy on historical data.
    
    Parameters
    ----------
    strategy : Strategy
        The trading strategy to backtest
    data : pd.DataFrame
        Historical price data with OHLC columns
    initial_capital : float, default=10000
        Starting capital for the backtest
        
    Returns
    -------
    dict
        Backtest results including returns, metrics, and trades
        
    Examples
    --------
    >>> results = backtest_strategy(my_strategy, price_data)
    >>> print(f"Total return: {results['total_return']:.2%}")
    """
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── conftest.py          # Shared test fixtures
├── test_strategies/     # Strategy tests
├── test_agents/         # Agent tests
├── test_utils/          # Utility function tests
└── test_integration/    # Integration tests
```

### Writing Tests

```python
import pytest
from src.strategies import TradingStrategy

class TestTradingStrategy:
    def test_strategy_initialization(self):
        """Test strategy creates with default parameters."""
        strategy = TradingStrategy()
        assert strategy.risk_level == 0.02
        
    def test_generate_signals(self, sample_price_data):
        """Test signal generation on sample data."""
        strategy = TradingStrategy()
        signals = strategy.generate_signals(sample_price_data)
        
        assert len(signals) == len(sample_price_data)
        assert all(signal in [-1, 0, 1] for signal in signals)
        
    @pytest.mark.parametrize("risk_level", [0.01, 0.02, 0.05])
    def test_risk_levels(self, risk_level, sample_price_data):
        """Test strategy with different risk levels."""
        strategy = TradingStrategy(risk_level=risk_level)
        # Test implementation...
```

### Test Categories

- **Unit Tests**: Test individual functions/methods
- **Integration Tests**: Test component interactions
- **Smoke Tests**: Basic functionality verification
- **Performance Tests**: Check for regressions

## 📚 Documentation

### Types of Documentation

1. **Code Comments**: Explain complex logic
2. **Docstrings**: API documentation
3. **README Updates**: User-facing changes
4. **Tutorials**: New feature guides
5. **API Docs**: Comprehensive reference

### Documentation Standards

- Keep explanations clear and concise
- Include practical examples
- Update docs with code changes
- Test code examples for accuracy

## 💡 Contribution Ideas

### 🔰 Beginner-Friendly

- Fix typos and improve documentation
- Add unit tests for existing functions
- Implement simple technical indicators
- Create example trading strategies

### 🚀 Intermediate

- Add new data sources or exchanges
- Implement risk management features
- Improve Telegram bot functionality
- Create visualization components

### 🏆 Advanced

- Develop new ML models or pipelines
- Implement advanced trading strategies
- Add real-time data streaming
- Performance optimization

## 🏷️ Commit Message Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(telegram): add portfolio balance command
fix(backtest): correct sharpe ratio calculation
docs(api): update trading strategy examples
test(ml): add tests for feature engineering
```

## 🤝 Community

### Getting Help

- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs and request features
- **Discord/Telegram**: Real-time community chat (if available)

### Recognition

Contributors are recognized in:
- Release notes for significant contributions
- Contributors section in README
- Hall of Fame for major contributions

## 📞 Contact

- **Maintainer**: [@cheafi](https://github.com/cheafi)
- **Email**: [Create an issue for direct contact]
- **Project**: [TradingAI Bot Repository](https://github.com/cheafi/TradingAI_Bot-main)

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

---

Thank you for contributing to TradingAI Bot! Your efforts help make quantitative trading more accessible to everyone. 🚀📈