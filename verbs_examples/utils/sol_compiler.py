import json
import os

from solcx import compile_files, install_solc


def compile_file(sol_file_path: str):
    """
    Compiles the solidity file in `sol_file_path` and saves
    the bytecode and the abi and the same path as the `sol_file_path`
    """
    install_solc(version="0.8.10")
    compiled_sol = compile_files(
        [sol_file_path], output_values=["abi", "bin"], solc_version="0.8.10"
    )
    contract_id, contract_interface = compiled_sol.popitem()
    dirname, contract_name = os.path.split(sol_file_path)
    contract_name = os.path.splitext(contract_name)[0]
    abi = contract_interface["abi"]
    bytecode = contract_interface["bin"]
    with open(os.path.join(dirname, contract_name + ".json"), "w") as f:
        json.dump({"bytecode": "0x" + bytecode}, f, indent=4)
    with open(os.path.join(dirname, contract_name + ".abi"), "w") as f:
        json.dump(abi, f)
