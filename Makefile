# Make setup the default target
.DEFAULT_GOAL := setup


setup:
	@echo "Running initial setup..."
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh