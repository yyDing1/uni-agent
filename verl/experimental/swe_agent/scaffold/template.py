# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


def convert_tools_to_description_text(tools: list[dict]) -> str:
    ret = ""
    for i, tool in enumerate(tools):
        assert tool["type"] == "function"
        fn = tool["function"]
        if i > 0:
            ret += "\n"
        ret += f"---- BEGIN FUNCTION #{i + 1}: {fn['name']} ----\n"
        ret += f"Description: {fn['description']}\n"

        if "parameters" in fn:
            ret += "Parameters:\n"
            properties = fn["parameters"].get("properties", {})
            required_params = set(fn["parameters"].get("required", []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                # Indicate required/optional in parentheses with type
                is_required = param_name in required_params
                param_status = "required" if is_required else "optional"
                param_type = param_info.get("type", "string")

                # Get parameter description
                desc = param_info.get("description", "No description provided")

                # Handle enum values if present
                if "enum" in param_info:
                    enum_values = ", ".join(f"`{v}`" for v in param_info["enum"])
                    desc += f"\nAllowed values: [{enum_values}]"

                ret += f"  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n"
        else:
            ret += "No parameters are required for this function.\n"

        ret += f"---- END FUNCTION #{i + 1} ----\n"
    return ret


def convert_tools_to_description_xml(tools: list[dict]) -> str:
    def render_extra_keys(json_dict, handled_keys):
        output = ""
        for k, v in json_dict.items():
            if k not in handled_keys:
                if isinstance(v, list) or isinstance(v, dict):
                    output += f"<{k}>{json.dumps(v, ensure_ascii=False)}</{k}>\n"
                else:
                    output += f"<{k}>{str(v)}</{k}>\n"
        return output

    ret = ""
    ret += "<tools>\n"
    for tool in tools:
        assert tool["type"] == "function"
        fn = tool["function"]
        ret += "<function>\n"
        ret += f"<name>{fn['name']}</name>\n"
        if "description" in fn:
            ret += f"<description>{fn['description'].strip()}</description>\n"
        ret += "<parameters>\n"
        parameters = fn.get("parameters")

        if isinstance(parameters, dict) and isinstance(parameters.get("properties"), dict):
            properties = parameters.get("properties", {})

            for param_name, param_fields in properties.items():
                ret += "<parameter>\n"
                ret += f"<name>{param_name}</name>\n"
                if "type" in param_fields:
                    ret += f"<type>{str(param_fields['type'])}</type>\n"
                if "description" in param_fields:
                    ret += f"<description>{param_fields['description'].strip()}</description>\n"
                ret += render_extra_keys(param_fields, ["name", "type", "description"])
                ret += "</parameter>\n"

        ret += render_extra_keys(parameters, ["type", "properties"])
        ret += render_extra_keys(fn, ["type", "name", "description", "parameters"])
        ret += "</parameters>\n"
        ret += "</function>\n"
    ret += "</tools>"
    return ret


class SWETemplateConfig(BaseModel):
    system_template: str = ""
    instance_template: str = ""
    tool_discription_type: str = Literal["xml", "text"]
    model_config = ConfigDict(extra="forbid")

    def get_system_prompt(self, tools: list[dict]) -> str:
        if self.tool_discription_type == "xml":
            tool_description = convert_tools_to_description_xml(tools)
        else:
            tool_description = convert_tools_to_description_text(tools)
        system_prompt = self.system_template.format(tool_description=tool_description.strip())
        return system_prompt
