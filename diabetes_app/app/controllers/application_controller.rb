class ApplicationController < ActionController::Base
  ML_API_URL = ENV.fetch("ML_API_URL", "http://localhost:5001")
end
