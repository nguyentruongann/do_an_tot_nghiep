FROM nginx:alpine
ARG API_BASE=http://localhost:8080
ENV API_BASE=${API_BASE}
COPY public /usr/share/nginx/html
RUN sed -i "s|__API_BASE__|${API_BASE}|g" /usr/share/nginx/html/index.html
EXPOSE 80