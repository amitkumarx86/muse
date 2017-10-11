# important libs
import pandas as pd
import datetime as dt
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

DATA_DIR = "/home/amit/Documents/hackathon/muse/data/"
FILE_NAME = "SOFT_GENd6661f5_new.csv"
MASTER_FILE = "ProductMaster404b8b3.xlsx"

# global variables
df = pd.DataFrame()
final_data = pd.DataFrame()
master_df = pd.DataFrame()
dataFrame = pd.DataFrame() 
need = pd.DataFrame()
# code for data for Saamu API 

# function for post request from app
def AI(data):
    temp = data
    field1 = temp.split("=")[0]
    field2 = temp.split("=")[1]

    # deciding function call
    if"hi" in field2:
        return "Hi Sir, How may I help you ?"
    if ("need" in field2 or "requirement" in field2 or "get" in field2) and "today" in field2:
        return what_need(0)
    if ("need" in field2 or "requirement" in field2 or "get" in field2) and "tomorrow" in field2:
        return what_need(1)
    if "predict" in field2 or "forecast" in field2:  
        category = field2.split("predict")
        if len(category) == 0:
            category = field2.split("forecast")
        return get_prediction(category[1].strip().lower())
    
    if ("sale" in field2 or "sell" in field2 or "sold" in field2 or "earn" in field2) and ("email" in field2 or "mail" in field2) and ("today" in field2) and ("complete" in field2):
        return complete_list_of_today_sale(email=True)

    if ("sale" in field2 or "sell" in field2 or "sold" in field2 or "earn" in field2) and ("today" in field2) and ("complete" in field2):
        return complete_list_of_today_sale()
    
    if ("sale" in field2 or "sell" in field2 or "sold" in field2 or "earn" in field2) and ("today" in field2):
        return sale_for_today()
    if "sale" in field2 or "sell" in field2 or "sold" in field2 or "earn" in field2:
        return most_selling_product(field2)

    if "bundle" in field2 or ("good" in field2 and "deal" in field2):
        return bundledProducts()
    if "thank" in field2 :
        return "You'r welcome"

    return "Perdon plz!"

# function for initial data loading
def loadData():
    global dataFrame
    global df
    global final_data
    global master_df
    # load raw data 
    print("Reading RW file...")
    df = pd.read_csv(DATA_DIR+FILE_NAME,delimiter="|")
    df.rename(columns={'CREATED_STAMP':'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # df['Date'] = df['Date'].dt.date

    # computing week day
    df['weekday'] = df['Date'].dt.weekday

    # retaining only those values which has integer barcode
    df = df[df['BARCODE'].astype(str).str.isdigit()]
    df.BARCODE = df.BARCODE.astype(str)
    print("Reading RW file done")
    
    # read master file
    print("Reading master file...")
    master_df = pd.read_excel(DATA_DIR+MASTER_FILE)
    master_df.head()
    master_df = master_df[master_df['BARCODE'].astype(str).str.isdigit()]
    master_df.BARCODE = master_df.BARCODE.astype(str)
    print("Reading master file done")

    print("Creating Final Data...")
    final_data = pd.merge(df, master_df, on="BARCODE")
    final_data.head()
    final_data.BASEPACK_DESC = final_data.BASEPACK_DESC.astype(str) 
    final_data['Date'] = pd.to_datetime(final_data['Date'], errors='coerce')
    final_data['Month'] = final_data['Date'].dt.month
    # see final data
    
    final_data = final_data[final_data.BASEPACK_DESC != "OTHERS"]
    final_data = final_data[final_data.BASEPACK_DESC != "0"]
    final_data['Date'] = final_data['Date'].dt.date
    final_data = final_data.sort_values(['BILLNO'])
    print("Creating FinalData done.")

    # dataFrame
    print("Creating DataFrame....")
    product_data = final_data.groupby(['Date','SUBCATEGORY_DESC','BASEPACK_DESC']).size()

    # print(product_data)
    # this will give following result
    # date | basepack_desc
    dataFrame = product_data.reset_index()
    
    # print(dataFrame.head())
    dataFrame.columns = ['Date','ItemCategory','Item','Quantity']
    dataFrame['Date'] = pd.to_datetime(dataFrame['Date'], errors='coerce')
    dataFrame['WeekDay'] = dataFrame['Date'].dt.weekday
    dataFrame.sort_values('Date')

    def f(row):
        dailyItems = list()
        dailyItems = ["BUNS & PAVS","BUTTER & CREAM","CAKES","CHEESE",
                      "CURD","PANEER","TOFU","OTHER SWEETS","VEG & FRUIT","YOGURT & LASSI","LADOOS","FISH"]
        if any(row['ItemCategory'] == x for x in dailyItems) :
            return 0
        return 1

    # dataFrame['type'] = dataFrame.apply(f, axis=1)
    temp = pd.DataFrame(np.random.randint(45,70,size=
                                        (len(dataFrame.Quantity), 1)), columns=list('A'))
    dataFrame['Stock'] = temp.A - dataFrame['Quantity']
    dataFrame = dataFrame.copy()
    print("Creating DataFrame done.")
    print("Saving dataFrame and final_data...")
    dataFrame.to_csv("dataFrame.csv",index=False)
    final_data.to_csv("final_data.csv",index=False)
    print("Saving dataFrame and final_data done")    
    # print(dataFrame.head())

# what do I need for today
def what_need(offset=0):
    # what do I need for today
    global dataFrame
    global df
    global final_data
    global master_df
    global need
    dataFrame = pd.read_csv("dataFrame.csv")
    
    
    # defind a threshold
    threshold = 70
    # print(dataFrame.head())
    # print(dataFrame[dataFrame.Stock < threshold][['ItemCategory','Item','WeekDay','Stock']].head(20))
    need[['ItemCategory','Item','WeekDay','Stock']] = dataFrame[dataFrame.Stock < threshold][['ItemCategory','Item','WeekDay','Stock']]
    # print(need.head())
    need = dataFrame.copy()
    # get today's day  number
    date = pd.to_datetime(dt.datetime.today()) + pd.DateOffset(days=offset)
    today = date.weekday()
    # clearn unknown category
    need = need[(need.WeekDay == today) & (need.ItemCategory != "Unknown") ]

    temp_1 = pd.DataFrame()
    temp_1['Stock'] = need['Stock'].abs()
    need.Stock = temp_1.Stock

    need = need.groupby(['ItemCategory','Item']).sum()
    need = need.reset_index()
    need = need.sort_values(['Stock'])
    # need = need[need.Stock < 11]
    
    need = need.head(20)
    need['need'] = 100 - need['Stock']
    need = need.sort_values('need',ascending=False)

    table = "<h5>Need for "+("Today" if offset == 0 else "Tomorrow")+"<h5>"
    table += "<table style='font-size:7px' id='product'>"
    table += "<th>ItemCategory</th><th>Item</th><th>Remain</th><th>Need</th></tr>"
    for index, row in need.iterrows():
        table += "<tr><td>"+str(row["ItemCategory"]).title()+"</td><td>"+str(row["Item"]).title()+"</td><td>"+str(row["Stock"])+"</td><td>"+str(100-row["Stock"])+"</tr>"
    table += "</table>"
    return table


def find_most_selling_product_week():
    final_data_new = final_data.groupby(['SUBCATEGORY_DESC','weekday']).size()
    final_data_new = final_data_new.reset_index()
    final_data_new.columns = ['Item','weekday','Quantity']
    return final_data_new.sort_values('Quantity',ascending=False)

def find_most_selling_product_month():
    final_data_new = final_data.groupby(['Month','SUBCATEGORY_DESC']).size()
    final_data_new = final_data_new.reset_index()
    final_data_new.columns = ['Month','Item','Quantity']
    return final_data_new.sort_values('Quantity',ascending=False)

def find_most_selling_product_overall():
    final_data_new = final_data.groupby('SUBCATEGORY_DESC').size()
    final_data_new = final_data_new.reset_index()
    final_data_new.columns = ['Item','Quantity']
    return final_data_new.sort_values('Quantity',ascending=False)

def most_selling_product(time="week"):
    global final_data
    final_data = pd.read_csv("final_data.csv")
    temp = pd.DataFrame()
    timeFrame = ""
    time = time.lower()
    if "month" in time:
        temp = find_most_selling_product_month()
        timeFrame = "Month"
    if "week" in time:
        temp = find_most_selling_product_week()
        timeFrame = "WeekDay"
    if "year" in time or "overall" in time:
        temp = find_most_selling_product_overall()
        timeFrame = "Year"
    else:
        temp = find_most_selling_product_week()
        timeFrame = "WeekDay"
    # set the return value
    table = ""
    if timeFrame != "WeekDay":
        table = "<h5>"+timeFrame+"wise Most selling Products:<h5>"
    else:
        table = "<h5>Weekwise Most selling Products:<h5>"

    table += "<table style='font-size:7px' id='product'>"
    if timeFrame == "Year":
        table += "<th>Item</th><th>Quantity</th>"
    else:
        table += "<th>Item</th><th>"+timeFrame+"</th><th>Quantity</th>"
    
    # only give top 20 products
    temp = temp.head(20)
    for index, row in temp.iterrows():
        if timeFrame == "Month":
            table += "<tr><td>"+str(row["Item"]).title()+"</td><td>"+str(row["Month"])+"</td><td>"+str(row["Quantity"])+"</td><td></tr>"
        if timeFrame == "WeekDay":
            table += "<tr><td>"+str(row["Item"]).title()+"</td><td>"+str(row["weekday"])+"</td><td>"+str(row["Quantity"])+"</td><td></tr>"
        if timeFrame == "Year":
            table += "<tr><td>"+str(row["Item"]).title()+"</td><td>"+str(row["Quantity"]).title()+"</td></tr>"

    table += "</table>"
    return table


def complete_list_of_today_sale(email=False):
    global final_data
    final_data = pd.read_csv("final_data.csv")
    from datetime import datetime
    startdate = datetime.now().strftime('%Y-%m-%d')
    date_val = pd.to_datetime(startdate) + pd.DateOffset(days=0)
    final_data.Date = final_data['Date'].astype(str)
    final_data['Date'] = pd.to_datetime(final_data['Date'], errors='coerce')
    # print(final_data.head())
    return_data = final_data[final_data.Date == "2017-01-01"][["Date","SUBCATEGORY_DESC","BASEPACK_DESC"]]
    return_data = return_data.groupby(['Date','SUBCATEGORY_DESC','BASEPACK_DESC']).size()

    return_data = return_data.reset_index()
    return_data.columns = ['Date','ItemCategory','Item','Count']
    return_data = return_data[return_data.ItemCategory != "Unknown"]
    return_data = return_data.sort_values('Count',ascending=False)
    
    total_sale = return_data.Count.sum()
    
    table = "<h5>Total Sale for Today: "+str(total_sale)+"<h5>"
    table += "<h6>Top Products for today<h6>"
    
    table += "<table style='font-size:7px' id='product'>"
    table += "<th>ItemCategory</th><th>Item</th><th>Quantity</th>"
    for index, row in return_data.iterrows():
        table += "<tr><td>"+str(row["ItemCategory"])+"</td><td>"+str(row["Item"]).title()+"</td><td>"+str(row["Count"])+"</td>  </tr>"
    table += "</table>"
    
    if email:
        import smtplib
        from email.mime.multipart import MIMEMultipart      
        from email.mime.text import MIMEText
        from email.utils import COMMASPACE
       

        user = 'amitkumarx86@gmail.com'  
        passw = '10987@412452~!Amit'

        smtp_host = 'smtp.gmail.com'
        smtp_port = 587
        server = smtplib.SMTP()
        server.connect(smtp_host,smtp_port)
        server.ehlo()
        server.starttls()
        server.login(user,passw)
        fromaddr = user
        tolist = "amitkumarx86@gmail.com"
        sub = "Report - Today's Sale"

        print("Data converted happening...")
        
        print("Data converted")
        try:
            msg = MIMEMultipart()
            msg['From'] = fromaddr
            msg['To'] = "amitkumarx86@gmail.com"
            msg['Subject'] = sub  
            msg.attach(MIMEText("Hi Amit, please find today's report attached."))
            attachment = MIMEText(table,'html')
            msg.attach(attachment)
            server.sendmail(user,tolist,msg.as_string())
            print("email send")
        except:
            print("Something went wrong..")

        table += "email sent."    

    return table
def sale_for_today():
    global final_data
    final_data = pd.read_csv("final_data.csv")
    from datetime import datetime
    startdate = datetime.now().strftime('%Y-%m-%d')
    date_val = pd.to_datetime(startdate) + pd.DateOffset(days=0)
    final_data.Date = final_data['Date'].astype(str)
    final_data['Date'] = pd.to_datetime(final_data['Date'], errors='coerce')
    # print(final_data.head())
    return_data = final_data[final_data.Date == "2017-01-01"][["Date","SUBCATEGORY_DESC","BASEPACK_DESC"]]
    return_data = return_data.groupby(['Date','SUBCATEGORY_DESC','BASEPACK_DESC']).size()

    return_data = return_data.reset_index()
    return_data.columns = ['Date','ItemCategory','Item','Count']
    return_data = return_data[return_data.ItemCategory != "Unknown"]
    return_data = return_data.sort_values('Count',ascending=False)
    
    total_sale = return_data.Count.sum()
    return_data = return_data.head(20)
    table = "<h5>Total Sale for Today: "+str(total_sale)+"<h5>"
    table += "<h6>Top Products for today<h6>"
    
    table += "<table style='font-size:7px' id='product'>"
    table += "<th>ItemCategory</th><th>Item</th><th>Quantity</th>"
    for index, row in return_data.iterrows():
        table += "<tr><td>"+str(row["ItemCategory"])+"</td><td>"+str(row["Item"]).title()+"</td><td>"+str(row["Count"])+"</td>  </tr>"
    table += "</table>"
    return table



# bundled products
def bundledProducts():      
    print("Good Deals function is called")

    # ---------------------------------------------------------------------
    # bundled product analysis                                            #
    # ---------------------------------------------------------------------

    # here begins final processing of bills
    # date | billno | item | quantity

    column_to_get = 'SUBCATEGORY_DESC'

    # read csv final data
    final_data = pd.read_csv("final_data.csv")

    bill_data = final_data.groupby(['BILLNO', column_to_get]).size()
    # print(bill_data.head())
    bill_data = bill_data.reset_index()

    prev_bill_no = bill_data['BILLNO'][0]
    # print(first_bill_no)

    market_basket_temp = [[]]
    i = 0
    temp = []
    for bill, item in zip(bill_data.BILLNO, bill_data[column_to_get]):
        if bill == prev_bill_no:
    #         market_basket[i] =  market_basket[i] + "," + item
            item = item.replace('&',',')
            item = item.replace('-',' ')
            if not any(item == x for x in ['HAIR','UNKNOWN','Hair','Unknown','PACKED FOOD']):
                temp.append(item)
        else:
            i = i + 1
            market_basket_temp.append(temp)
            temp = []
            item = item.replace('&',',')
            item = item.replace('-',' ')
            if not any(item == x for x in ['HAIR','UNKNOWN','Hair','Unknown','PACKED FOOD']):
                temp.append(item)
            prev_bill_no = bill

    market_basket = []
    for item in market_basket_temp:
        l = ":".join(item)
        if len(l) > 0:
            market_basket.append(l)

    temp_df = pd.DataFrame()
    temp_df['Products'] = market_basket

    # now get unique rows
    market_basket_dataFrame = pd.DataFrame()
    market_basket_dataFrame['Products'] = temp_df.Products.unique()
    # market_basket_dataFrame_new['Products'] = market_basket_dataFrame['Products'].unique()
    market_basket_dataFrame.to_csv("Products.csv",index=False)
#         len(market_basket_dataFrame.Products)
    
    
    support = 100

    # how many products we want together to see
    itemset_size = 3

    itemsets_dct = main("Products.csv", support, itemset_size)

    bundled_product = dict()
    for item, count in itemsets_dct.items():
        temp = []
        for i in item:
            if i != ',':
                i = i.replace("\"",'')
                temp.append(i)
        bundled_product[tuple(temp)] = count
    
    bundled_product_new = sorted(bundled_product.items(), key=lambda x: x[1], reverse = True)
    # len(bundled_product.items())
    bundled_product_new = sorted(bundled_product.items(), key=lambda x: x[1], reverse = True)
    # len(bundled_product.items())
    result1 = []
    result2 = []
    for i in bundled_product_new:
        result1.append(str(list(i)[0]).replace('(','').replace(')','').replace('\'','').replace("'",''))
        result2.append(list(i)[1])

    bundled_dataFrame = pd.DataFrame()
    bundled_dataFrame['Product'] = result1
    bundled_dataFrame['Freq'] = result2

    table = "<h5>Good Deals Are:<h5>"
    table += "<table style='font-size:7px' id='product'>"
    table += "<th>Items</th><th>Freq</th>"
    
    # only give top 20 products
    bundled_dataFrame = bundled_dataFrame.sample(n=10)
    bundled_dataFrame = bundled_dataFrame.sort_values('Freq',ascending=False)
    for index, row in bundled_dataFrame.iterrows():
        table += "<tr><td>"+str(row["Product"]).title()+"</td><td>"+str(row["Freq"])+"</td></tr>"
    table += "</table>"

    return table
    
def main(file_location, support, itemset_size):
    import time
    from collections import defaultdict
    candidate_dct = defaultdict(lambda: 0)
    for i in range(itemset_size):
        now = time.time()
        candidate_dct = data_pass(file_location, support, pass_nbr=i+1, candidate_dct=candidate_dct)
        print ("pass number %i took %f and found %i candidates" % (i+1, time.time()-now, len(candidate_dct)))
    return candidate_dct

def data_pass(file_location, support, pass_nbr, candidate_dct):
    with open(file_location, 'r') as f:
        for line in f:
            item_lst = line.split(":")     
            candidate_dct = update_candidates(item_lst, candidate_dct, pass_nbr)

    candidate_dct = clear_items(candidate_dct, support, pass_nbr)

    return candidate_dct

def update_candidates(item_lst, candidate_dct, pass_nbr):
    from itertools import combinations
    if pass_nbr==1:
        for item in item_lst:
            candidate_dct[(item,)] += 1
    else:
        frequent_items_set = set()
        for item_tuple in combinations(sorted(item_lst), pass_nbr-1):    
            if item_tuple in candidate_dct:
                frequent_items_set.update(item_tuple)

        for item_set in combinations(sorted(frequent_items_set), pass_nbr):
            candidate_dct[item_set]+=1

    return candidate_dct

def clear_items(candidate_dct, support, pass_nbr):
    for item_tuple, cnt in list(candidate_dct.items()):
        if cnt<support or len(item_tuple)<pass_nbr:
            del candidate_dct[item_tuple]
    return candidate_dct



# predict the data for the category
def predict_product_quantity(subCategory, plotOption=False):
    
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt

    # getting holiday list
    holidays = pd.read_csv(DATA_DIR+'holiday.csv',header=None,delimiter=",")
    holidays.columns = ['Date','Festival']
    holidays['Date'] = pd.to_datetime(holidays['Date'], errors='coerce')


    dataFrame = pd.read_csv("dataFrame.csv")
    dataFrame.ItemCategory = dataFrame['ItemCategory'].str.lower()
    items = dataFrame[dataFrame.ItemCategory == subCategory][['Date','Quantity','Item']]
    print(items.head())
    # group_item = items.groupby('Date').size()
    # group_item = group_item.reset_index()
    group_item = pd.DataFrame()
    group_item[['Date','Quantity']] = items[['Date','Quantity']]
    print(group_item.head())

    # group_item['WeekDay'] = group_item['Date'].dt.weekday
    # group_item.sort_values('Quantity', ascending=False).head(5)
    # group_item = group_item.groupby(['Date','WeekDay']).sum()
    # group_item = group_item.reset_index()
    group_item['Date'] = pd.to_datetime(group_item['Date'], errors='coerce')
    group_item['Month'] = group_item['Date'].dt.month
    group_item['WeekDay'] = group_item['Date'].dt.weekday

    # holidays.head()

    result = []
    fest = []
    from datetime import datetime
    for date in group_item['Date']:
        for v,f in zip(holidays['Date'],holidays['Festival']):
            if date <= v:
                delta = v - date
                result.append(delta.days)
                fest.append(f)
                break

    group_item['Days_remainging'] = result
    group_item['Festival'] = fest
    group_item_new = group_item
    # group_item_new.drop('Date', axis = 1)

    prediction_data = pd.DataFrame()
    prediction_data[['ds','y']] = group_item[['Date','Quantity']] 

#     group_item.head(5)
    # learning machine learning model
    

    lm = LinearRegression()
    X = group_item.drop(['Quantity','Date','Festival'], axis = 1)
    Y = group_item.Quantity
    lm.fit(X,Y)
    
    # predict for comming 7 days
    no_of_days_predict = 7;

     
    startdate = datetime.now().strftime('%Y-%m-%d')
    result = []

    # prepare date for visualization purpose
    predicted_data = pd.DataFrame()

    fest = ""

    # set the product type
    # Type = 1
    # dailyItems = ["BUNS & PAVS","BUTTER & CREAM","CAKES","CHEESE",
    #               "CURD","PANEER","TOFU","OTHER SWEETS","VEG & FRUIT","YOGURT & LASSI","LADOOS","FISH"]
    # if any(subCategory == x for x in dailyItems):
    #     Type = 0

    for i in range(no_of_days_predict):
        date_val = pd.to_datetime(startdate) + pd.DateOffset(days=i)
        weekday = date_val.weekday()
        month = date_val.month
        holiday_remaning = 0
        for v,f in zip(holidays['Date'], holidays['Festival']):
            if date_val <= v:
                holiday_remaning = ((v - date_val).days)*2
                fest = f
                break;
        X = [[]]
        X[0].append(weekday)
        X[0].append(month)
        X[0].append(holiday_remaning)


        # predict results added
        predicted_val = int(np.ceil(lm.predict(X)))
        if(predicted_val < 0):
            predicted_val = 1

        df2 = pd.DataFrame([[date_val, predicted_val, fest,holiday_remaning]], 
                           columns=list(['Date','Quantity','Festival','Days_Remaining']))
        predicted_data = predicted_data.append(df2, ignore_index=True)
        result.append(predicted_val)


    # festival days calculation
    predicted_data['Days_Remaining'] = predicted_data['Days_Remaining']/2
        
    
    
    predicted_data['WeekDay'] = predicted_data['Date'].dt.weekday_name
    predicted_data = predicted_data[['Date','WeekDay','Festival','Days_Remaining','Quantity']]
    
    if plotOption:
        # plot for trending 
        plt.figure(figsize = (10,20))
        prediction_data.plot(x="ds",y="y",kind='line',figsize=(8,4),grid=True,title='Actual Sale of Product '+subCategory)

        # plot the chart for prediction

        type_of_chart = 'line'
        yLim=(0,np.max(predicted_data['Quantity'])*1.3)
        
        predicted_data.plot(x='WeekDay',y='Quantity',kind = type_of_chart,figsize=(8,4),grid=True,
                            title='Prediction of Products '+subCategory,ylim=yLim)
        plt.show()
         
    # how much to maintain
    
    return predicted_data, str(predicted_data.Quantity.sum())


# predicting the value of some product for coming week
def get_prediction(data):
    result,value = predict_product_quantity(data)
    print(result.head())
    table = "<h5>Prediction for "+data+"</h5>" 
    table += "<table style='font-size:7px' id='product'>"
    table += "<th>Festival</th><th>Days_Remaining</th><th>Quantity</th></tr>"
    for index, row in result.iterrows():
        table += "<tr><td>"+str(row["Festival"]).title()+"</td><td>"+str(row["Days_Remaining"])+"</td><td>"+str(row["Quantity"])+"</tr>"
    table += "</table>"+"<h5>Total Quantity to put : "+str(value)+"</h5>"
    print(table)
    return table
# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):           
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "https://*")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Access-Control-Allow-Headers", "dataType, accept, authorization") 
        self.send_header('Content-type',    'text/html')                                    
        self.end_headers()              
        self.wfile.write(bytes("<html><body>Hello world!</body></html>", "utf8"))

    def do_POST(self):   
        # Send response status code
        self.send_response(200)
        
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        
        #  Send headers
        self.send_header('Content-type','text/html')
        self.send_header('Access-Control-Allow-Origin', 'https://172.16.80.181:8080/')                
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")   
        self.end_headers()
        
        # Send message back to client
        message = AI(data.decode("utf-8"))
        print(message)
        # Write content as utf-8 data
        self.wfile.write(bytes(message, "utf8"))
        return

     
    
def run():
        print('starting server...')
        import ssl
        # Server settings
        # Choose port 8080, for port 80, which is normally used for a http server, you need root access
        server_address = ('192.168.43.202', 8080)
        httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
       
        httpd.socket = ssl.wrap_socket(httpd.socket, certfile='/home/amit/Documents/cert.pem', server_side=True)
       
        # first call to load initial data
        print("loading initial data")
        # loadData()
        print("loading initial data done :)")
        
        print('running server...')
        httpd.serve_forever()

        
run()
# sale_for_today()
# # loadData()
# # what_need_today()
# loadData()
# get_prediction("HEALTHY & DIGESTIVE BISCUITS".lower())