FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    exiftool \
    libarchive-zip-perl \
    poppler-utils \
    ffmpeg \
    python3.12 \
    python3.12-venv \
    python3-pip \
    xvfb \
    fluxbox \
    x11vnc \
    novnc \
    websockify \
    supervisor \
    socat \
    fonts-crosextra-carlito \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fonts-dejavu-mono \
    fonts-liberation \
    fonts-noto-color-emoji \
    fonts-noto-mono \
    fonts-freefont-ttf \
    fonts-urw-base35 \
    fonts-wqy-zenhei \
    fonts-droid-fallback \
    libgtk-3-0t64 \
    libnss3 \
    libasound2t64 \
    libatk-bridge2.0-0t64 \
    libatk1.0-0t64 \
    libdrm2 \
    libgbm1 \
    libx11-6 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libxss1 \
    libxshmfence1 \
    libglib2.0-0t64 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libcairo2 \
    libxcb-dri3-0 \
    libxcb-glx0 \
    libxkbcommon0 \
    libdrm-amdgpu1 \
    libdrm-intel1 \
    fonts-liberation-sans-narrow \
    libwoff1 \
    libfontconfig1 \
    libfreetype6 \
    libreoffice-impress \
    libreoffice-writer \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG NODE_VERSION=22.23.2
ARG GH_VERSION=2.97.0
ARG BIRD_VERSION=0.8.0

RUN curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" -o /tmp/node.tar.gz \
    && tar -xzf /tmp/node.tar.gz -C /usr/local --strip-components=1 \
    && rm -f /tmp/node.tar.gz \
    && node --version && npm --version

RUN curl -fsSL "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" -o /tmp/gh.tar.gz \
    && tar -xzf /tmp/gh.tar.gz -C /tmp \
    && mv /tmp/gh_${GH_VERSION}_linux_amd64/bin/gh /usr/local/bin/gh \
    && rm -rf /tmp/gh_${GH_VERSION}_linux_amd64 /tmp/gh.tar.gz \
    && gh --version

RUN npm install -g @steipete/bird@${BIRD_VERSION} \
    && bird --version

RUN pip install --no-cache-dir cloakbrowser==0.6.3 playwright==1.61.0 && cloakbrowser install

RUN mkdir -p /root/log /app/data/resources /app/app /app/tests

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY app/ /app/app/
COPY tests/__init__.py /app/tests/__init__.py
COPY tests/app /app/tests/app
RUN pip install --no-cache-dir -e ".[test]"

COPY supervisor.conf /etc/supervisor/conf.d/supervisor.conf

EXPOSE 80 5900 6080 9222

CMD ["bash", "-c", "mkdir -p /root/log/supervisor && /usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisor.conf"]
