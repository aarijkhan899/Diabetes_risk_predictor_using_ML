class PredictionsController < ApplicationController
  FEATURE_LABELS = {
    "Pregnancies"             => "Pregnancies (count)",
    "Glucose"                 => "Glucose (mg/dL)",
    "BloodPressure"           => "Blood Pressure (mm Hg)",
    "SkinThickness"           => "Skin Thickness (mm)",
    "Insulin"                 => "Insulin (mu U/mL)",
    "BMI"                     => "BMI (kg/m²)",
    "DiabetesPedigreeFunction" => "Diabetes Pedigree Function",
    "Age"                     => "Age (years)"
  }.freeze

  FEATURE_RANGES = {
    "Pregnancies"             => { min: 0,    max: 20,    step: 1,      placeholder: "e.g. 2" },
    "Glucose"                 => { min: 0,    max: 250,   step: 1,      placeholder: "e.g. 120" },
    "BloodPressure"           => { min: 0,    max: 150,   step: 1,      placeholder: "e.g. 70" },
    "SkinThickness"           => { min: 0,    max: 100,   step: 1,      placeholder: "e.g. 20" },
    "Insulin"                 => { min: 0,    max: 900,   step: 1,      placeholder: "e.g. 80" },
    "BMI"                     => { min: 0,    max: 70,    step: 0.1,    placeholder: "e.g. 28.5" },
    "DiabetesPedigreeFunction" => { min: 0,    max: 2.5,   step: 0.001,  placeholder: "e.g. 0.627" },
    "Age"                     => { min: 18,   max: 120,   step: 1,      placeholder: "e.g. 35" }
  }.freeze

  FEATURE_TOOLTIPS = {
    "Pregnancies"             => "Number of times pregnant.",
    "Glucose"                 => "Plasma glucose concentration at 2 hours in an oral glucose tolerance test.",
    "BloodPressure"           => "Diastolic blood pressure in mm Hg.",
    "SkinThickness"           => "Triceps skin fold thickness in mm.",
    "Insulin"                 => "2-hour serum insulin in mu U/mL.",
    "BMI"                     => "Body mass index — weight (kg) divided by height (m) squared.",
    "DiabetesPedigreeFunction" => "A function that scores likelihood of diabetes based on family history.",
    "Age"                     => "Age of the patient in years."
  }.freeze

  EXAMPLE_PATIENT = {
    "Pregnancies"             => "6",
    "Glucose"                 => "148",
    "BloodPressure"           => "72",
    "SkinThickness"           => "35",
    "Insulin"                 => "0",
    "BMI"                     => "33.6",
    "DiabetesPedigreeFunction" => "0.627",
    "Age"                     => "50"
  }.freeze

  def index
    @feature_labels  = FEATURE_LABELS
    @feature_ranges  = FEATURE_RANGES
    @feature_tooltips = FEATURE_TOOLTIPS
    @example_patient = EXAMPLE_PATIENT
  end

  def predict
    input_params = extract_input_params
    errors = validate_input(input_params)

    if errors.any?
      flash[:alert] = "Please correct the following: #{errors.join(', ')}"
      redirect_to root_path and return
    end

    begin
      response = HTTParty.post(
        "#{ML_API_URL}/predict",
        body:    input_params.to_json,
        headers: { "Content-Type" => "application/json", "Accept" => "application/json" },
        timeout: 30
      )

      if response.success?
        @result        = response.parsed_response
        @input_params  = input_params
        @feature_labels = FEATURE_LABELS
        render :result
      else
        error_body = response.parsed_response rescue {}
        error_msg  = error_body.is_a?(Hash) ? error_body["error"] : "Unknown API error"
        flash[:alert] = "Prediction API error (HTTP #{response.code}): #{error_msg}"
        redirect_to root_path
      end

    rescue HTTParty::Error, Net::OpenTimeout, Net::ReadTimeout, SocketError => e
      flash[:alert] = (
        "Could not connect to the prediction service. " \
        "Please ensure the Flask API is running (python ml/api.py). " \
        "Error: #{e.class}"
      )
      redirect_to root_path
    rescue StandardError => e
      Rails.logger.error("PredictionsController#predict unexpected error: #{e.message}\n#{e.backtrace.first(5).join("\n")}")
      flash[:alert] = "An unexpected error occurred. Please try again."
      redirect_to root_path
    end
  end

  def about
    # Attempt to fetch model info from the API for display on the about page
    begin
      response = HTTParty.get(
        "#{ML_API_URL}/model_info",
        headers: { "Accept" => "application/json" },
        timeout: 5
      )
      @model_info = response.success? ? response.parsed_response : {}
    rescue StandardError
      @model_info = {}
    end
  end

  def health_check
    begin
      response = HTTParty.get(
        "#{ML_API_URL}/health",
        headers: { "Accept" => "application/json" },
        timeout: 5
      )
      if response.success?
        api_status = response.parsed_response
        render json: {
          rails_status: "ok",
          ml_api_status: api_status
        }, status: :ok
      else
        render json: {
          rails_status: "ok",
          ml_api_status: "unreachable",
          ml_api_http_code: response.code
        }, status: :service_unavailable
      end
    rescue StandardError => e
      render json: {
        rails_status: "ok",
        ml_api_status: "unreachable",
        error: e.class.to_s
      }, status: :service_unavailable
    end
  end

  private

  def extract_input_params
    permitted = params.permit(
      :Pregnancies, :Glucose, :BloodPressure,
      :SkinThickness, :Insulin, :BMI,
      :DiabetesPedigreeFunction, :Age
    )
    permitted.to_h
  end

  def validate_input(input)
    errors = []
    FEATURE_RANGES.each do |feature, range|
      value = input[feature]
      if value.blank?
        errors << "#{FEATURE_LABELS[feature]} is required"
        next
      end
      begin
        num = Float(value)
        if num < range[:min] || num > range[:max]
          errors << "#{FEATURE_LABELS[feature]} must be between #{range[:min]} and #{range[:max]}"
        end
      rescue ArgumentError
        errors << "#{FEATURE_LABELS[feature]} must be a number"
      end
    end
    errors
  end
end
