COPY ./ ./
RUN chmod +x scripts/ci/entrypoint.sh
ENTRYPOINT ["sh", "scripts/ci/entrypoint.sh"]
