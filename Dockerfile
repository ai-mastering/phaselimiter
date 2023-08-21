FROM contribu/buildenv_docker:bionic

ENV APP_ROOT /app
ENV RAILS_ENV production
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /etc/phaselimiter/bin:/etc/phaselimiter/script:$PATH
WORKDIR $APP_ROOT

RUN (echo -e "\n\n\n" | ssh-keygen -t rsa) \
    && curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl \
    && chmod a+rx /usr/local/bin/youtube-dl

RUN apt-get update && apt-get install -y \
  ruby-dev libxml2 libxml2-dev zlib1g-dev

#  --without test developmentはつけない。
# rspecの実行時と差分があるといろいろ考慮しないといけなくていやなので
COPY Gemfile $APP_ROOT
COPY Gemfile.lock $APP_ROOT
RUN bundle install --jobs=8 --retry=3 \
 || bundle install --retry=3 \
 || bundle install --retry=3

COPY . $APP_ROOT

RUN ( \
        mv $APP_ROOT/phaselimiter /etc/phaselimiter \
        && cd /etc/phaselimiter \
        && chmod +x bin/* \
        && chmod +x script/audio_detector \
        && pyenv exec pipenv install \
    )

EXPOSE  3000
CMD ['rails', 's']
