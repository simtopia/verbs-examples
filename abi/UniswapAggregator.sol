// SPDX-License-Identifier: BUSL-1.1
pragma solidity 0.8.10;

interface IUniswap {
    function slot0() external view returns (uint160, int24, uint16, uint16, uint16, uint8, bool);
}

contract UniswapAggregator {
  address private _uniswap_pool_address;
  bool private _order_tokens_ab;
  // Use this to unscale the price, but also keep 8 decimal places
  uint256 private constant divisor = uint256(2 ** 96);

  constructor(address uniswapPoolAddress, address tokenA, address tokenB) {
    _uniswap_pool_address = uniswapPoolAddress;
    _order_tokens_ab = tokenA < tokenB? true : false;
  }

  function latestAnswer() external view returns (int256) {
    (uint256 sqrt_price_x96,,,,,,) = IUniswap(_uniswap_pool_address).slot0();
    uint256 scaled_price = _order_tokens_ab ? (10 ** 4 * sqrt_price_x96 / divisor) ** 2 : (10 ** 4 * divisor / sqrt_price_x96) ** 2;
    return int256(scaled_price);
  }

  function getTokenType() external pure returns (uint256) {
    return 1;
  }

  function decimals() external pure returns (uint8) {
    return 8;
  }
}
