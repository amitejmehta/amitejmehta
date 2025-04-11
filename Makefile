# Make setup the default target
.DEFAULT_GOAL := setup


setup:
	@echo "Running initial setup..."
	@chmod +x setup.sh
	@./setup.sh