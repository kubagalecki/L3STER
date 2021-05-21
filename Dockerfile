
WORKDIR /CI
COPY . /CI
RUN chmod +x scripts/ci/entrypoint.sh
ENTRYPOINT ["sh", "scripts/ci/entrypoint.sh"]
