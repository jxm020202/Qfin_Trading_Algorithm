from typing import Dict
import numpy
import pandas as pd

class TradingAlgorithm:

    def __init__(self):
        self.positions: Dict[str, int] = {} # self.positions is a dictionary that will keep track of your position in each product 
                                            # E.g. {"ABC": 2, "XYZ": -5, ...}
                                            # This will get automatically updated after each call to getOrders()

        #
        # TODO: Initialise any other variables that you want to use to keep track of things between getOrders() calls here
        #
        #roll_bid_window = {"SOBER":[], "UEC":[]}
        #roll_ask_window = {"SOBER":[], "UEC":[]}
        self.shifted_mid = {"SOBER":[], "UEC":[]}
        #rolling_bid = {"SOBER":[], "UEC":[]}
        #rolling_ask = {"SOBER":[], "UEC":[]}
        self.rolling_mean = {"SOBER":0, "UEC":0}
        self.threshold = {"SOBER":4, "UEC":0.5}

        self.outtime = {"SOBER":[], "UEC":[]}
        self.outprice = {"SOBER":[], "UEC":[]}
        self.outtrade = {"SOBER":[], "UEC":[]}
        self.outquant = {"SOBER":[], "UEC":[]}

    def addOutput(self, time, stock, price, trade, quant):
        self.outtime[stock].append(time)
        self.outprice[stock].append(price)
        self.outtrade[stock].append(trade)
        self.outquant[stock].append(quant)

    # This method will be called every timestamp with information about the new best bid and best ask for each product
    def getOrders(self, current_data: Dict[str, Dict[str, float]], order_data: Dict[str, int]) -> Dict[str, int]:
        # current_data is a dictionary that holds the current timestamp, best bid and best ask for each product
        # E.g. {"ABC": {"Timestamp": 134, "Bid": 34, "Ask" 38}, "XYZ": {"Timestamp": 134, "Bid": 1034, "Ask": 1038}, ...}
        
        # order_data is a dictionary that holds the quantity you will order for each product in this current timestamp
        # Intially the quantity for each product is set to 0 (i.e. no buy or sell orders will be sent if order_data is returned as it is)
        # To buy ABC for quantity x -> order_data["ABC"] = x (This will buy from the current best ask)
        # To sell ABC for quantity x -> order_data["ABC"] = -x (This will sell to the current best bid)
        
        #
        # TODO: Process new data and populate order_data here
        #
        for stock in current_data:
            #roll_bid_window[stock].append(current_data[stock]["Bid"])
            #roll_ask_window[stock].append(current_data[stock]["Ask"])
            self.shifted_mid[stock].append((current_data[stock]["Bid"] + current_data[stock]["Ask"])/2)
            if len(self.shifted_mid[stock]) > 20:
                #roll_bid_window.pop(0)
                #roll_ask_window.pop(0)
                self.shifted_mid[stock].pop(0)
                self.rolling_mean[stock] = numpy.average(self.shifted_mid[stock])

                if (self.shifted_mid[stock][0] - self.rolling_mean[stock]) < -(self.threshold[stock] * 1.5):
                    order_data[stock] = -5
                    print("Selling 5 of " + stock + " at timestamp " + str(current_data[stock]["Timestamp"]) + ".\n")
                    self.addOutput(current_data[stock]["Timestamp"], stock, current_data[stock]["Bid"], "Sell", 5)
                elif (self.shifted_mid[stock][0] - self.rolling_mean[stock]) < -(self.threshold[stock] * 1):
                    order_data[stock] = -1
                    print("Selling 1 of " + stock + " at timestamp " + str(current_data[stock]["Timestamp"]) + ".\n")
                    self.addOutput(current_data[stock]["Timestamp"], stock, current_data[stock]["Bid"], "Sell", 1)
                elif (self.shifted_mid[stock][0] - self.rolling_mean[stock]) > (self.threshold[stock] * 1.5):
                    order_data[stock] = 5
                    print("Buying 5 of " + stock + " at timestamp " + str(current_data[stock]["Timestamp"]) + ".\n")
                    self.addOutput(current_data[stock]["Timestamp"], stock, current_data[stock]["Ask"], "Buy", 5)
                elif (self.shifted_mid[stock][0] - self.rolling_mean[stock]) > (self.threshold[stock] * 1):
                    order_data[stock] = 1
                    print("Buying 1 of " + stock + " at timestamp " + str(current_data[stock]["Timestamp"]) + ".\n")
                    self.addOutput(current_data[stock]["Timestamp"], stock, current_data[stock]["Ask"], "Buy", 1)

        


        return order_data
    
    # You may create more methods to help process data but ensure they are inside this class
    


# Leave this stuff as it is
team_algorithm = TradingAlgorithm()

def getOrders(current_data, positions):
    team_algorithm.positions = positions
    order_data = {product: 0 for product in current_data}
    return team_algorithm.getOrders(current_data, order_data)

def printToExcel():
    with pd.ExcelWriter('orders.xlsx') as writer:
        for stock in team_algorithm.outtime:
            data = {"Timeframe": team_algorithm.outtime[stock], "Price": team_algorithm.outprice[stock], "Trade Type": team_algorithm.outtrade[stock], "Quantity": team_algorithm.outquant[stock]}
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=stock, index=False, engine='openpyxl')
    print("Orders stored in 'orders.xlsx' file.\n")
