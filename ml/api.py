        shap_values = explainer.shap_values(X_scaled)
        # For binary classifiers, shap_values may be list[2 arrays]
        if isinstance(shap_values, list):
            sv = shap_values[1][0].tolist()   # positive class
        else:
            sv = shap_values[0].tolist()
        shap_dict = dict(zip(FEATURES, sv))
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")
        shap_dict = {}

    # Plain-language guidance
    label = "Diabetic" if pred == 1 else "Non-Diabetic"
    if pred == 1:
        top_feature = (
            max(shap_dict, key=lambda k: abs(shap_dict[k]))
            if shap_dict else "Glucose"
        )
        guidance = (
            f"This patient presents elevated diabetes risk. "
            f"The most influential clinical factor is {top_feature}. "
            "Immediate clinical review is recommended."
        )
    else:
        guidance = (
            "This patient currently shows low diabetes risk based on available "
