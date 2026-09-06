"""
TRACE-ID — Contract Deployment Script
Compiles and deploys the EvidenceRegistry Solidity contract to Polygon Amoy.

Usage:
    python -m app.blockchain.deploy

Requirements:
    - POLYGON_RPC_URL set in .env
    - PRIVATE_KEY set in .env (dedicated testnet wallet)
    - Sufficient MATIC on Polygon Amoy for gas
      (Faucet: https://faucet.polygon.technology/ — select Amoy)

After deployment:
    - CONTRACT_ADDRESS is printed and saved to app/blockchain/deployed_address.txt
    - Copy the address to your .env file: CONTRACT_ADDRESS=<address>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as: python -m app.blockchain.deploy
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from app import config  # noqa: E402 (after sys.path fix)


def compile_contract() -> tuple[str, list]:
    """
    Compile the EvidenceRegistry.sol contract using py-solc-x.

    Returns:
        Tuple of (bytecode_hex_string, abi_list).

    Raises:
        RuntimeError: If compilation fails.
    """
    try:
        from solcx import compile_source, install_solc
    except ImportError as exc:
        raise RuntimeError(
            "py-solc-x not installed. Run: pip install py-solc-x"
        ) from exc

    contract_path = Path(__file__).parent / "contract.sol"
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract not found: {contract_path}")

    source = contract_path.read_text(encoding="utf-8")

    # Extract required solc version from pragma
    import re
    match = re.search(r"pragma solidity\s+\^?(\d+\.\d+\.\d+)", source)
    if not match:
        raise RuntimeError("Could not determine Solidity version from pragma")
    solc_version = match.group(1)

    print(f"[deploy] Solidity version: {solc_version}")
    print("[deploy] Installing solc compiler (if not cached)...")

    try:
        install_solc(solc_version, show_progress=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to install solc {solc_version}: {exc}") from exc

    print("[deploy] Compiling EvidenceRegistry.sol...")
    try:
        compiled = compile_source(
            source,
            solc_version=solc_version,
            output_values=["abi", "bin"],
        )
    except Exception as exc:
        raise RuntimeError(f"Solidity compilation failed: {exc}") from exc

    # The key format is "<stdin>:ContractName"
    contract_key = None
    for key in compiled:
        if "EvidenceRegistry" in key:
            contract_key = key
            break

    if not contract_key:
        raise RuntimeError(
            f"EvidenceRegistry contract not found in compiled output. Keys: {list(compiled.keys())}"
        )

    abi = compiled[contract_key]["abi"]
    bytecode = compiled[contract_key]["bin"]

    print(f"[deploy] Compilation successful. Bytecode length: {len(bytecode)} chars")

    # Save ABI for future use
    abi_path = Path(__file__).parent / "compiled_contract.json"
    with open(abi_path, "w", encoding="utf-8") as f:
        json.dump({"abi": abi, "bytecode": bytecode}, f, indent=2)
    print(f"[deploy] ABI saved: {abi_path}")

    return bytecode, abi


def deploy_contract(bytecode: str, abi: list) -> str:
    """
    Deploy the compiled contract to Polygon Amoy.

    Args:
        bytecode: Hex bytecode string (without 0x prefix).
        abi:      Contract ABI list.

    Returns:
        Deployed contract address (checksummed).

    Raises:
        RuntimeError: On connection or deployment failure.
    """
    if not config.POLYGON_RPC_URL:
        raise RuntimeError("POLYGON_RPC_URL is not set in .env")
    if not config.PRIVATE_KEY:
        raise RuntimeError(
            "PRIVATE_KEY is not set in .env\n"
            "Use a dedicated testnet wallet — NEVER use a wallet with real funds."
        )

    try:
        from web3 import Web3
        from web3.middleware import ExtraDataToPOAMiddleware
    except ImportError as exc:
        raise RuntimeError("web3 not installed. Run: pip install web3") from exc

    print(f"[deploy] Connecting to: {config.POLYGON_RPC_URL}")
    w3 = Web3(Web3.HTTPProvider(config.POLYGON_RPC_URL))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    if not w3.is_connected():
        raise RuntimeError(
            f"Cannot connect to Polygon Amoy: {config.POLYGON_RPC_URL}\n"
            "Check your internet connection and RPC URL."
        )

    chain_id = w3.eth.chain_id
    network_names = {
        80002: "Polygon Amoy Testnet",
        11155111: "Ethereum Sepolia Testnet",
        1: "Ethereum Mainnet",
        137: "Polygon Mainnet",
    }
    explorer_bases = {
        80002: "https://amoy.polygonscan.com",
        11155111: "https://sepolia.etherscan.io",
        1: "https://etherscan.io",
        137: "https://polygonscan.com",
    }
    network_name = network_names.get(chain_id, f"EVM Chain (ID: {chain_id})")
    explorer_base = explorer_bases.get(chain_id, "https://etherscan.io")
    token_symbol = "POL" if chain_id in (80002, 137) else "ETH"
    print(f"[deploy] Connected to {network_name}. Chain ID: {chain_id}")

    # Derive wallet
    pk = config.PRIVATE_KEY.lstrip("0x")
    account = w3.eth.account.from_key(pk)
    wallet = account.address

    balance_wei = w3.eth.get_balance(wallet)
    balance_eth = w3.from_wei(balance_wei, "ether")
    print(f"[deploy] Wallet: {wallet}")
    print(f"[deploy] Balance: {balance_eth:.6f} {token_symbol}")

    if balance_eth < 0.0005:
        raise RuntimeError(
            f"Insufficient {token_symbol} balance: {balance_eth:.6f}\n"
            "Please fund your wallet with testnet tokens."
        )

    # Build deployment transaction
    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(wallet)
    gas_price = w3.eth.gas_price

    print("[deploy] Estimating gas...")
    try:
        gas_estimate = contract.constructor().estimate_gas({"from": wallet})
        gas_limit = int(gas_estimate * 1.3)
    except Exception as exc:
        raise RuntimeError(f"Gas estimation failed: {exc}") from exc

    print(f"[deploy] Gas estimate: {gas_estimate} | Limit: {gas_limit}")

    tx = contract.constructor().build_transaction(
        {
            "from": wallet,
            "nonce": nonce,
            "gas": gas_limit,
            "gasPrice": gas_price,
        }
    )

    # Sign and deploy
    signed = w3.eth.account.sign_transaction(tx, pk)
    print("[deploy] Sending deployment transaction...")
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"[deploy] Transaction: {tx_hash.hex()}")
    print("[deploy] Waiting for confirmation...")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt["status"] != 1:
        raise RuntimeError("Contract deployment transaction reverted")

    contract_address = receipt["contractAddress"]
    print(f"\n{'='*60}")
    print("CONTRACT DEPLOYED SUCCESSFULLY")
    print(f"  Network:          {network_name} (chain_id={chain_id})")
    print(f"  Contract Address: {contract_address}")
    print(f"  Transaction:      {tx_hash.hex()}")
    print(f"  Block:            {receipt['blockNumber']}")
    print(f"  Gas Used:         {receipt['gasUsed']}")
    print(f"  Explorer:         {explorer_base}/address/{contract_address}")
    print(f"{'='*60}")
    print(f"\nNEXT STEP: Add to your .env file:")
    print(f"  CONTRACT_ADDRESS={contract_address}")

    # Save address to file
    addr_path = Path(__file__).parent / "deployed_address.txt"
    addr_path.write_text(contract_address + "\n", encoding="utf-8")
    print(f"\n[deploy] Address saved to: {addr_path}")

    return contract_address


def main() -> None:
    print("=" * 60)
    print("TRACE-ID — EvidenceRegistry Contract Deployment")
    print("Network: Polygon Amoy Testnet")
    print("=" * 60)

    config_warnings = config.validate_config()
    # Only check blockchain-related warnings for deploy
    bc_warnings = [w for w in config_warnings if "POLYGON" in w or "PRIVATE" in w]
    if bc_warnings:
        for w in bc_warnings:
            print(f"[warn] {w}")

    try:
        bytecode, abi = compile_contract()
        deploy_contract(bytecode, abi)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
