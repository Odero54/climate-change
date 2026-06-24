"""Tests for registry.py — USE_CASE_REGISTRY, ModelOption, UseCaseInfo, get_use_case_info."""

import pytest

from climate_change.registry import USE_CASE_REGISTRY, ModelOption, UseCaseInfo, get_use_case_info

EXPECTED_MODULES = {"drought", "flood", "food_security", "disease", "land_degradation"}


class TestUseCaseRegistry:
    def test_all_expected_modules_present(self):
        assert set(USE_CASE_REGISTRY.keys()) == EXPECTED_MODULES

    def test_each_entry_is_use_case_info(self):
        for module_id, info in USE_CASE_REGISTRY.items():
            assert isinstance(info, UseCaseInfo), f"{module_id} is not a UseCaseInfo"

    def test_each_info_has_id_matching_key(self):
        for module_id, info in USE_CASE_REGISTRY.items():
            assert info.id == module_id

    def test_each_info_has_non_empty_name(self):
        for info in USE_CASE_REGISTRY.values():
            assert info.name

    def test_each_info_has_model_options(self):
        for info in USE_CASE_REGISTRY.values():
            assert len(info.model_options) > 0

    def test_default_model_is_in_model_options(self):
        for info in USE_CASE_REGISTRY.values():
            option_ids = {opt.id for opt in info.model_options}
            assert info.default_model in option_ids, (
                f"{info.id}: default_model '{info.default_model}' not in options {option_ids}"
            )

    def test_min_months_positive(self):
        for info in USE_CASE_REGISTRY.values():
            assert info.min_months > 0

    def test_max_years_positive(self):
        for info in USE_CASE_REGISTRY.values():
            assert info.max_years > 0

    def test_drought_has_cdi_in_dependent_variable(self):
        assert "CDI" in USE_CASE_REGISTRY["drought"].dependent_variable

    def test_flood_has_risk_class_in_dependent_variable(self):
        assert "risk" in USE_CASE_REGISTRY["flood"].dependent_variable.lower()


class TestModelOption:
    def test_fields_stored(self):
        opt = ModelOption(id="rf", label="Random Forest", recommended=True, note="fast")
        assert opt.id == "rf"
        assert opt.label == "Random Forest"
        assert opt.recommended is True
        assert opt.note == "fast"

    def test_is_frozen(self):
        opt = ModelOption(id="rf", label="RF", recommended=False, note="")
        with pytest.raises(AttributeError):
            opt.id = "changed"


class TestGetUseCaseInfo:
    def test_returns_correct_info(self):
        info = get_use_case_info("drought")
        assert info.id == "drought"

    @pytest.mark.parametrize("module_id", list(EXPECTED_MODULES))
    def test_returns_info_for_all_modules(self, module_id):
        info = get_use_case_info(module_id)
        assert info.id == module_id

    def test_unknown_module_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown module"):
            get_use_case_info("nonexistent_module")
