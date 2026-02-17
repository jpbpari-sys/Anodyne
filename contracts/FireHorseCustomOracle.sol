// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@chainlink/contracts/src/v0.8/ChainlinkClient.sol";

contract FireHorseCustomOracle is ChainlinkClient {
    using Chainlink for Chainlink.Request;

    uint256 public confidence;
    bytes32 public jobId;
    uint256 public fee;
    address public oracle;

    event SignalRequested(bytes32 indexed requestId);
    event ConfidenceUpdated(bytes32 indexed requestId, uint256 confidence);

    constructor(address linkToken, address oracleAddress, bytes32 chainlinkJobId, uint256 requestFee) {
        setChainlinkToken(linkToken);
        setChainlinkOracle(oracleAddress);

        oracle = oracleAddress;
        jobId = chainlinkJobId;
        fee = requestFee;
    }

    function requestSignal() external returns (bytes32 requestId) {
        Chainlink.Request memory req = buildChainlinkRequest(
            jobId,
            address(this),
            this.fulfill.selector
        );

        req.add("get", "http://your-oracle-host/api/signal");
        req.add("path", "confidence");

        requestId = sendChainlinkRequestTo(oracle, req, fee);
        emit SignalRequested(requestId);
    }

    function fulfill(bytes32 requestId, uint256 updatedConfidence)
        external
        recordChainlinkFulfillment(requestId)
    {
        confidence = updatedConfidence;
        emit ConfidenceUpdated(requestId, updatedConfidence);
    }
}
