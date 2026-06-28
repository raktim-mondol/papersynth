# Contributing to PaperSynth

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone and setup
git clone https://github.com/raktim-mondol/papersynth.git
cd papersynth
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Add your API keys to .env
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Add docstrings to all public methods
- Keep modules focused — one responsibility per file

## Making Changes

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feat/my-feature`
3. **Make** your changes
4. **Test** your changes: `pytest tests/ -v`
5. **Commit**: `git commit -m "feat: add my feature"`
6. **Push**: `git push origin feat/my-feature`
7. **Open** a Pull Request

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation changes
- `refactor:` — Code restructuring without behavior change
- `test:` — Adding or updating tests
- `chore:` — Maintenance tasks

## Areas for Contribution

### High Priority
- [ ] PubMed/MEDLINE data source integration
- [ ] PDF paper parsing (for full-text analysis)
- [ ] Interactive visualization of the knowledge graph
- [ ] Better methodology keyword extraction (NER-based)

### Medium Priority
- [ ] Support for other LLMs (OpenAI, Anthropic, local models)
- [ ] Batch processing (multiple queries)
- [ ] Export to LaTeX/BibTeX
- [ ] Web UI (Streamlit or Gradio)

### Low Priority
- [ ] Docker containerization
- [ ] GitHub Actions CI/CD
- [ ] Benchmark suite for gap detection quality
- [ ] Multi-language support

## Reporting Issues

Please use the [GitHub Issues](https://github.com/raktim-mondol/papersynth/issues) page. Include:

1. Python version
2. Operating system
3. Steps to reproduce
4. Expected vs actual behavior
5. Error output (if any)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
