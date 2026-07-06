# Development workflow

1. Create a branch from `main`.
2. Make one logically coherent change.
3. Add or update tests where appropriate.
4. Run:

```bash
python -m compileall src python examples tests
pytest
```

5. Open a pull request describing the scientific or software motivation.

## Research-code expectations

- Do not silently change the physical model.
- Record changes to simulation parameters.
- Keep schema changes documented in `docs/data_schema.md`.
- Do not commit large generated datasets or trained model binaries.
- Add validation when introducing new preprocessing transformations.
- Verify surrogate-optimised controls using the numerical simulator or experiment before making physical-performance claims.
