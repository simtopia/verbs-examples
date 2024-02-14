def mint_and_approve_dai(
    env,
    dai_abi,
    dai_address: bytes,
    contract_approved_address: bytes,
    dai_admin_address: bytes,
    recipient: bytes,
    amount: int,
):

    dai_abi.mint.execute(
        address=dai_address,
        sender=dai_admin_address,
        env=env,
        args=[recipient, amount],
    )

    dai_abi.approve.execute(
        sender=recipient,
        address=dai_address,
        env=env,
        args=[contract_approved_address, amount],
    )


def mint_and_approve_weth(
    env,
    weth_abi,
    weth_address: bytes,
    recipient: bytes,
    contract_approved_address: bytes,
    amount: int,
):
    weth_abi.deposit.execute(
        address=weth_address,
        args=[],
        env=env,
        sender=recipient,
        value=amount,
    )

    weth_abi.approve.execute(
        sender=recipient,
        address=weth_address,
        env=env,
        args=[contract_approved_address, amount],
    )
