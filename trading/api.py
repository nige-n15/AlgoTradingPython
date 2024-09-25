import time
import urllib.parse
import hashlib
import hmac
import base64
import requests
import os
from typing import Dict, Any


class KrakenAPI:
    def __init__(self):
        with open(("env/keys"), "r") as f:
            lines = f.read().splitlines()
            api_key = lines[0]
            api_sec = lines[1]

        self.api_url = "https://api.kraken.com"
        self.api_key = api_key
        self.api_sec = api_sec

        if not self.api_key or not self.api_sec:
            raise ValueError("API key and secret must be set in env/keys")

    def _get_kraken_signature(self, urlpath: str, data: Dict[str, Any]) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_sec), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()

    def _kraken_request(self, url_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._get_kraken_signature(url_path, data)
        }
        resp = requests.post((self.api_url + url_path), headers=headers, data=data)
        if resp.status_code != 200:
            raise Exception(f"API request failed with status {resp.status_code}: {resp.text}")
        return resp.json()

    def _create_order(self, order_type: str, trade_type: str, volume: str, pair: str, price: str = None) -> Dict[
        str, Any]:
        data = {
            "nonce": str(int(time.time() * 1000000)),
            "ordertype": order_type,
            "type": trade_type,
            "volume": volume,
            "pair": pair,
        }
        if price:
            data["price"] = price
        return self._kraken_request("/0/private/AddOrder", data)

    def market_buy(self, buy_amt: str, pair: str = "XBTUSD") -> Dict[str, Any]:
        return self._create_order("market", "buy", buy_amt, pair)

    def market_sell(self, sell_amt: str, pair: str = "XBTUSD") -> Dict[str, Any]:
        return self._create_order("market", "sell", sell_amt, pair)

    def limit_buy(self, buy_amt: str, price: str, pair: str = "XBTUSD") -> Dict[str, Any]:
        return self._create_order("limit", "buy", buy_amt, pair, price)

    def limit_sell(self, sell_amt: str, price: str, pair: str = "XBTUSD") -> Dict[str, Any]:
        return self._create_order("limit", "sell", sell_amt, pair, price)

    def cancel_all(self) -> Dict[str, Any]:
        data = {"nonce": str(int(time.time() * 1000000))}
        return self._kraken_request("/0/private/CancelAll", data)


# Usage example:
if __name__ == "__main__":
    kraken = KrakenAPI()

    # Example: Place a market buy order
    # response = kraken.market_buy("0.001")
    # print(response)

    # Example: Get open orders
    # open_orders = kraken._kraken_request("/0/private/OpenOrders", {"nonce": str(int(time.time() * 1000000))})
    # print(open_orders)