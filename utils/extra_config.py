import os
import yaml
import folder_paths
import logging

def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    for section_name, section_value in config.items():
        if section_value is None:
            continue

        base_path = None
        is_default = False

        if isinstance(section_value, dict):
            section_conf = dict(section_value)
            if "base_path" in section_conf and section_conf["base_path"] is not None:
                base_path = section_conf.pop("base_path")
                base_path = os.path.expandvars(os.path.expanduser(str(base_path)))
                if not os.path.isabs(base_path):
                    base_path = os.path.abspath(os.path.join(yaml_dir, base_path))
            if "is_default" in section_conf:
                is_default = bool(section_conf.pop("is_default"))
            entries = section_conf
        else:
            entries = {section_name: section_value}

        for model_name, raw_paths in entries.items():
            if raw_paths is None:
                continue

            if isinstance(raw_paths, (list, tuple, set)):
                path_candidates = [str(path).strip() for path in raw_paths if str(path).strip()]
            else:
                path_candidates = [line.strip() for line in str(raw_paths).splitlines() if line.strip()]

            for candidate in path_candidates:
                full_path = candidate
                if base_path:
                    full_path = os.path.join(base_path, full_path)
                elif not os.path.isabs(full_path):
                    full_path = os.path.abspath(os.path.join(yaml_dir, full_path))
                normalized_path = os.path.normpath(full_path)
                logging.info("Adding extra search path {} {}".format(model_name, normalized_path))
                folder_paths.add_model_folder_path(model_name, normalized_path, is_default)
