Rails.application.routes.draw do
  root "predictions#index"
  post "/predict", to: "predictions#predict"
  get  "/about",   to: "predictions#about"
  get  "/health",  to: "predictions#health_check"
end
