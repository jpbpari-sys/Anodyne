// SPDX-License-Identifier: MIT
pragma solidity ^0.8.7;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

contract FireHorsePriceConsumer {
    AggregatorV3Interface internal priceFeed;

    constructor() {
        // Plug Chainlink ETH/USD feed on Sepolia
        priceFeed = AggregatorV3Interface(0x694AA1769357215DE4FAC081bf1f309aDC325306);
    }

    function getLatestPrice() public view returns (int) {
        (, int price,,,) = priceFeed.latestRoundData(); // Returns price * 10^8
        return price;
    }

    // Apply to your oracle: Trigger action if price > threshold
    function checkSignal() external view returns (string memory) {
        int price = getLatestPrice();
        // Integrate your off-chain signal logic here (or via custom oracle)
        return price > 3000 * 10**8 ? "LONG" : "SHORT";
    }
}
