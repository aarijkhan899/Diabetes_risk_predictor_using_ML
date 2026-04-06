module ApplicationHelper
  def risk_badge_class(prediction)
    prediction == 1 ? "badge-danger" : "badge-success"
  end

  def confidence_bar_class(confidence)
    if confidence >= 80
      "bg-danger"
    elsif confidence >= 60
      "bg-warning"
    else
      "bg-success"
    end
  end
end
