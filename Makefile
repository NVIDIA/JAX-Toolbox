.PHONY: docs-dev docs-restart docs-wait docs-check docs-preview

docs-dev:
	npm run docs:dev

docs-restart:
	-pkill -f "fern docs dev" 2>/dev/null || true
	sleep 1
	npm run docs:dev

# Poll until the dev server responds — run this after backgrounding docs-restart
docs-wait:
	@echo "Waiting for http://localhost:3000 ..."
	@until curl -sf http://localhost:3000 >/dev/null 2>&1; do sleep 3; done
	@echo "Server ready."

docs-check:
	npm run docs:check

docs-preview:
	npm run docs:preview
