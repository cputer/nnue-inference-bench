# Convenience targets (placeholder)

.PHONY: smoke
smoke:
	python tools/smoke_cpu.py
	python bench/runner.py
