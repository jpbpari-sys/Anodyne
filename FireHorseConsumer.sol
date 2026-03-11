// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

contract FireHorseConsumer {
    AggregatorV3Interface internal oracle;
    address public owner;

    constructor(address _oracle) {
        oracle = AggregatorV3Interface(_oracle);
        owner = msg.sender;
    }

    function getSignal() public view returns (string memory) {
        (, int256 answer, , , ) = oracle.latestRoundData();
        return answer == 1 ? "LONG" : "SHORT";
    }

    function autoTrade() external {
        require(msg.sender == owner, "Only owner");

        string memory sig = getSignal();
        sig;
        // TODO: Integrate DEX execution logic based on signal.
    }
}
