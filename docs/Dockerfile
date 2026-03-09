FROM python:3.11-slim

WORKDIR /docs

# Install MkDocs and Material theme
RUN pip install --no-cache-dir \
    mkdocs==1.5.3 \
    mkdocs-material==9.4.14

# Copy documentation source
COPY mkdocs.yml .
COPY source/ ./source/

# Expose port for MkDocs
EXPOSE 8000

# Serve documentation
CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8000"]

