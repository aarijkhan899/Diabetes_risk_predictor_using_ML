require "active_support/core_ext/integer/time"

Rails.application.configure do
  config.enable_reloading = false
  config.eager_load = true
  config.consider_all_requests_local = false
  config.force_ssl = false
  config.log_level = :info
  config.cache_store = :memory_store
  config.active_support.report_deprecations = false

  config.public_file_server.enabled = ENV["RAILS_SERVE_STATIC_FILES"].present?

  config.hosts << "localhost"
  config.hosts << "127.0.0.1"
  config.hosts << "rails_app"
end
