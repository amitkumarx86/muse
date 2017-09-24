import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
         
from itertools import combinations
import time
from collections import defaultdict


from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
 
import random


    
###################################################################################################################
# global variable
rawFile = ""
masterFile = ""
master_df = pd.DataFrame()
df = pd.DataFrame()
final_data = pd.DataFrame()
bill_data_temp = pd.DataFrame()
summaryWidget=""
billWidget=""
dataFrame = pd.DataFrame()

from PyQt5.QtWidgets import QWidget, QListWidget, QListWidgetItem, QLabel, QApplication, QDialog


# class for generating table view
class PandasModel(QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None
        
        
class App(QMainWindow):
    
    # global variable
    global rawFile
    global masterFile
    global df
    global master_df
    global final_data
    global bill_data_temp
    global summaryWidget
    global billWidget
    global dataFrame
    
###################################################################################################################
    
    def predict_product_quantity(self,plotOption=True):

        import pandas as pd
        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt
        
        # getting holiday list
        self.holidays = pd.read_csv('../data/holiday.csv',header=None,delimiter=",")
        self.holidays.columns = ['Date','Festival']
        self.holidays['Date'] = pd.to_datetime(self.holidays['Date'], errors='coerce')
        
        
        subCategory = self.category
        items = self.dataFrame[self.dataFrame.ItemCategory == subCategory][['Date','Quantity','Item','type']]

        # group_item = items.groupby('Date').size()
        # group_item = group_item.reset_index()
        group_item = pd.DataFrame()
        group_item[['Date','Quantity','type']] = items[['Date','Quantity','type']]


        # group_item['WeekDay'] = group_item['Date'].dt.weekday
        # group_item.sort_values('Quantity', ascending=False).head(5)
        # group_item = group_item.groupby(['Date','WeekDay']).sum()
        # group_item = group_item.reset_index()
        group_item['Month'] = group_item['Date'].dt.month
        group_item['WeekDay'] = group_item['Date'].dt.weekday

        # holidays.head()

        result = []
        fest = []
        from datetime import datetime
        for date in group_item['Date']:
            for v,f in zip(self.holidays['Date'],self.holidays['Festival']):
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
        X = group_item.drop(['Quantity','Date','Festival','type'], axis = 1)
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
        Type = 1
        dailyItems = ["BUNS & PAVS","BUTTER & CREAM","CAKES","CHEESE",
                      "CURD","PANEER","TOFU","OTHER SWEETS","VEG & FRUIT","YOGURT & LASSI","LADOOS","FISH"]
        if any(subCategory == x for x in dailyItems):
            Type = 0

        for i in range(no_of_days_predict):
            date_val = pd.to_datetime(startdate) + pd.DateOffset(days=i)
            weekday = date_val.weekday()
            month = date_val.month
            holiday_remaning = 0
            for v,f in zip(self.holidays['Date'], self.holidays['Festival']):
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

            df2 = pd.DataFrame([[date_val, predicted_val,Type, fest,holiday_remaning]], 
                               columns=list(['Date','Quantity','type','Festival','Days_Remaining']))
            predicted_data = predicted_data.append(df2, ignore_index=True)
            result.append(predicted_val)


        # festival days calculation
        predicted_data['Days_Remaining'] = predicted_data['Days_Remaining']/2

        # how much to maintain
        if predicted_data.type.unique == 0:
            print("Quantity to put in inventory(daily) for "+subCategory+" is : " +str(predicted_data.Quantity.max))
        else:
            print("Quantity to put in inventory(Weekly) for "+subCategory+" is : " +str(predicted_data.Quantity.sum()))

        predicted_data['WeekDay'] = predicted_data['Date'].dt.weekday_name
        predicted_data = predicted_data[['Date','WeekDay','Festival','Days_Remaining','type','Quantity']]

#         if plotOption:
#             # plot for trending 
#             data = prediction_data['y'].tolist()
            
            
#             plt.figure(figsize = (10,20))
#             prediction_data.plot(x="ds",y="y",kind='line',figsize=(8,4),grid=True,title='Actual Sale of Product '+subCategory)
            
#             # plot the chart for prediction

#             type_of_chart = 'line'
#             yLim=(0,np.max(predicted_data['Quantity'])*1.3)

#             predicted_data.plot(x='WeekDay',y='Quantity',kind = type_of_chart,figsize=(8,4),grid=True,
#                                 title='Prediction of Products '+subCategory,ylim=yLim)
#             plt.show()
#             data = predicted_data['Quantity'].tolist()
#             ex1 = Graph()
            

        return predicted_data
    
##################################################################################################################
    def fileMenu(self):
        
        file = self.bar.addMenu("File")
        openRawFile = QAction("Open Raw File",self)
        openRawFile.setShortcut("Ctrl+O")
        openRawFile.triggered.connect(self.openRawFile)
        file.addAction(openRawFile)

        openMasterFile = QAction("Open Master File",self)
        openMasterFile.setShortcut("Ctrl+O")
        openMasterFile.triggered.connect(self.openMasterFile)
        file.addAction(openMasterFile)

        quit = QAction("Quit",self) 
        quit.setShortcut("Ctrl+Q")
        quit.triggered.connect(self.quitFunc)
        file.addAction(quit)
#         file.triggered[QAction].connect(self.processtrigger)
        
    def mainWindow(self):
        # buttons on left plane
        w = QWidget()
        b1 = QPushButton("")
        b1.setFixedHeight(100)
        b1.setFixedWidth(100)      
        b1.setStyleSheet('background-color:#4472c4;color: white;text-align: center;    text-decoration: none;  display: inline-block;    font-size: 16px;    margin: 4px 2px;    cursor: pointer;')
        b1.setIcon(QIcon(QPixmap("./img/summary_name.png")))
        b1.setIconSize(QSize(70, 70))  

        b2 = QPushButton("")
        b2.setFixedHeight(100)
        b2.setFixedWidth(100)      
        b2.setStyleSheet('background-color:#4472c4;color: white;text-align: center;    text-decoration: none;    display: inline-block;    font-size: 16px;       cursor: pointer;')
        b2.setIcon(QIcon(QPixmap("./img/bill_name.png")))
        b2.setIconSize(QSize(65, 65)) 

        b3 = QPushButton("")
        b3.setFixedHeight(100)
        b3.setFixedWidth(100)      
        b3.setStyleSheet('background-color:#4472c4;color: white;text-align: center;    text-decoration: none;   display: inline-block;    font-size: 16px;    cursor: pointer;')
        b3.setIcon(QIcon(QPixmap("./img/predict_name.png")))
        b3.setIconSize(QSize(70, 70)) 

        b4 = QPushButton("")
        b4.setFixedHeight(100)
        b4.setFixedWidth(100)
        b4.setStyleSheet('background-color:#4472c4;color: white;text-align: center;    text-decoration: none;   display: inline-block;    font-size: 16px;       cursor: pointer;')
        b4.setIcon(QIcon(QPixmap("./img/good_deals_name.png")))
        b4.setIconSize(QSize(80, 80)) 
        
        b5 = QPushButton("")
        b5.setFixedHeight(100)
        b5.setFixedWidth(100)
        b5.setStyleSheet('background-color:#4472c4;color: white;text-align: center;    text-decoration: none;    display: inline-block;    font-size: 16px;       cursor: pointer;')
        b5.setIcon(QIcon(QPixmap("./img/analytics-name.png")))
        b5.setIconSize(QSize(80, 80)) 

        vbox = QVBoxLayout()
        vbox1 = QVBoxLayout()      
        
        vbox1.addWidget(b2)
        vbox1.addWidget(b3)
        vbox1.addWidget(b4) 
        vbox1.addWidget(b1)
        vbox1.addWidget(b5)
        
        w.setLayout(vbox1)
        w.setStyleSheet('background-color:#4d5358;')

        left = QFrame()
        left.setFrameShape(QFrame.StyledPanel)
        left.setLayout(vbox)

        # for splitter
        hbox = QHBoxLayout(self)   


#         self.right.setFrameShape(QFrame.StyledPanel)

        splitter1 = QSplitter(Qt.Horizontal)
        splitter1.addWidget(w)
        splitter1.addWidget(self.right)
        splitter1.setSizes([1,100000])

        # top frame
        splitter2 = QSplitter(Qt.Vertical)

        w1 = QWidget()
        w1.setStyleSheet('background-image: url(./img/TOP.jpg);background-repeat: no-repeat;')
        self.right.setFrameShape(QFrame.StyledPanel)
        
        splitter2.addWidget(w1)
        splitter2.addWidget(splitter1)
        splitter2.setSizes([100,100,100])      
        hbox.addWidget(splitter2)

        self.setLayout(hbox)
        QApplication.setStyle(QStyleFactory.create('Cleanlooks'))
        self.setGeometry(50, 300, 300, 200)
        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(hbox)

        ########################################################################################
        # add functions on buttons
        ########################################################################################
        b1.clicked.connect(self.summary)
        b2.clicked.connect(self.bill)
        b3.clicked.connect(self.predict)
        b4.clicked.connect(self.gooddeals)
        b5.clicked.connect(self.analytics)
        
#         self.analytics()
        
#         self.bill()
        
    def processtrigger(self,q):
      # print q.text()+" is triggered"
        if q.text() == "Open Raw File": 
            self.rawFile, _ = QFileDialog.getOpenFileName()
        if q.text() == "Open Master File":
            self.masterFile, _ = QFileDialog.getOpenFileName()
        if q.text() == "Quit":
             sys.exit()
    
    def openRawFile(self):
        self.rawFile, _ = QFileDialog.getOpenFileName()
        
    def openMasterFile(self):
        self.masterFile, _ = QFileDialog.getOpenFileName()
        
    def quitFunc(self):
        sys.exit()
        
    def createLoading(self):
        # Create loading widget
        label = QLabel("Loading...")
#         pixmap = QPixmap('./img/giphy.gif')
#         label.setPixmap(pixmap)
#         self.resize(pixmap.width(),pixmap.height())
        vbox1 = QHBoxLayout()
        vbox1.addWidget(label)
        self.right.setLayout(vbox1)
    
    def getData(self):
        print("Raw Data is getting retrieved...")
        self.df = pd.read_csv(self.rawFile,delimiter="|")
        print("Raw Data is retrieved...")
        
        self.df.rename(columns={'CREATED_STAMP':'Date'}, inplace=True)
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')

        # computing week day
        self.df['weekday'] = self.df['Date'].dt.weekday

        # retaining only those values which has integer barcode
        self.df = self.df[self.df['BARCODE'].astype(str).str.isdigit()]
        self.df.BARCODE = self.df.BARCODE.astype(str)

        # read master file
        print("Master Data is getting retrieved...")
        self.master_df = pd.read_excel(self.masterFile)
        print("Master Data is retrieved...")
        self.master_df = self.master_df[self.master_df['BARCODE'].astype(str).str.isdigit()]
        self.master_df.BARCODE = self.master_df.BARCODE.astype(str)
        
        print("Data is getting consolidated...")
        self.final_data = pd.merge(self.df, self.master_df, on="BARCODE")
        print("Data is getting consolidation done.")
        
        self.final_data.BASEPACK_DESC = self.final_data.BASEPACK_DESC.astype(str) 
        self.final_data['Date'] = pd.to_datetime(self.final_data['Date'], errors='coerce')
        self.final_data['Month'] = self.final_data['Date'].dt.month
        self.final_data['Date'] = pd.to_datetime(self.final_data['Date'], errors='coerce')
        self.final_data = self.final_data[self.final_data.BASEPACK_DESC != "OTHERS"]
        self.final_data = self.final_data[self.final_data.BASEPACK_DESC != "0"]
        self.final_data['Date'] = self.final_data['Date'].dt.date
        self.final_data.sort_values(['BILLNO'])
        print(self.final_data.head())
        
    def summary(self):
        print("Summary function is called")
        # remove other widgets
        try:
            for i in reversed(range(self.commonLayout.count())): 
                self.commonLayout.itemAt(i).widget().deleteLater()
        except:
            print("error")
        
        self.summaryWidget = QWidget()
        
        try:
            
#             self.rawFile = "/home/amit/Documents/hackathon/muse/data/SOFT_GENd6661f5_new1.csv"
#             self.masterFile = "/home/amit/Documents/hackathon/muse/data/ProductMaster404b8b3.xlsx"
            print(self.rawFile)
            print(self.masterFile)
        except:
            QMessageBox.about(self, "Warning Box", "First Select raw & master files.")
            return
        
        # processing for raw data
        print("Collecting data initial data")
        self.getData()
        print("We got final data")
        
        print("Getting table view...")
        model = PandasModel(self.final_data.head(50)[['POS_Application_Name','STOREID','BILLNO','BARCODE','Date','CATEGORY_DESC','SUBCATEGORY_DESC','BASEPACK','BASEPACK_DESC']])
        self.tableView = QTableView(self)
        print("Setting table view...")
        
        
        self.tableView.setModel(model)
        
        self.vboxTable1 = QVBoxLayout()
        self.vboxTable1.addWidget(self.tableView)      
        
        self.summaryWidget.setLayout(self.vboxTable1)
        self.commonLayout.addWidget(self.summaryWidget)
        # assign tableView
        
        
        self.right.setLayout(self.commonLayout)
    
        
    def generateBill(self):
        pass
    
    def getDataFrame(self):
        product_data = self.final_data.groupby(['Date','SUBCATEGORY_DESC','BASEPACK_DESC']).size()
        # this will give following result
        # date | basepack_desc
        self.dataFrame = product_data.reset_index()
        self.dataFrame.columns = ['Date','ItemCategory','Item','Quantity']
        self.dataFrame['Date'] = pd.to_datetime(self.dataFrame['Date'], errors='coerce')
        self.dataFrame['WeekDay'] = self.dataFrame['Date'].dt.weekday
        self.dataFrame.sort_values('Date')

        def f(row):
            dailyItems = list()
            dailyItems = ["BUNS & PAVS","BUTTER & CREAM","CAKES","CHEESE",
                          "CURD","PANEER","TOFU","OTHER SWEETS","VEG & FRUIT","YOGURT & LASSI","LADOOS","FISH"]
            if any(row['ItemCategory'] == x for x in dailyItems) :
                return 0
            return 1

        self.dataFrame['type'] = self.dataFrame.apply(f, axis=1)
        return self.dataFrame
        
    # this method will for coming 7 days
    def productTypeClick(self,item):
        self.category = str(item.text())
        QMessageBox.about(self, "Information", "Prediction for "+self.category+" for next 7 days is getting calculated...")
        self.dataFrame = self.getDataFrame()
        
        print(self.dataFrame.head())
        temp =  self.predict_product_quantity(True)
        box = QVBoxLayout()
        figure = plt.figure()
        canvas = FigureCanvas(figure)
        data = temp['Quantity'].tolist()
        
        ax = figure.add_subplot(111)
        ax.plot_date(temp['Date'], temp['Quantity'], markerfacecolor='CornflowerBlue', markeredgecolor='white')
        ax.hold(False)
        ax.plot(data, '*-')
        canvas.draw()
        
        model = PandasModel(temp)
        self.tableViewPredict = QTableView(self)
        self.tableViewPredict.setModel(model)
        box.addWidget(self.tableViewPredict)
        box.addWidget(canvas)
        self.tableViewPredictWidget.setLayout(box)
              
    # predict function for product type
    def predict(self):
        print("Predict function is called")
        # remove other widgets
        try:
            for i in reversed(range(self.commonLayout.count())): 
                self.commonLayout.itemAt(i).widget().deleteLater()
        except :
            print("error")
        # plug them into vertical layout
        
        self.predictWidget = QWidget()
        
        layout1 = QHBoxLayout()
        print("Setting Product Type table...")
        self.productType = QListWidget()
        
        self.productType.itemDoubleClicked.connect(self.productTypeClick)
        item = QListWidgetItem("Product Type")
        
        for value in self.final_data[self.final_data['SUBCATEGORY_DESC'] != "nan"]['SUBCATEGORY_DESC'].unique():
            item = QListWidgetItem(str(value))
            self.productType.addItem(item)
        print("Setting Product Type table done.")
        
        print("Setting Prediction Table...")
        
        
        
        print("Setting Product Type table done.")
        
        
        layout1.addWidget(self.productType)
        
        self.tableViewPredictWidget = QWidget()
        
        layout1.addWidget(self.tableViewPredictWidget)
        
        
        self.predictWidget.setLayout(layout1)
        self.commonLayout.addWidget(self.predictWidget)
        self.right.setLayout(self.commonLayout)
        
    def gooddeals(self):      
        print("Good Deals function is called")

        for i in reversed(range(self.commonLayout.count())): 
            self.commonLayout.itemAt(i).widget().deleteLater()
        
        # ---------------------------------------------------------------------
        # bundled product analysis                                            #
        # ---------------------------------------------------------------------

        # here begins final processing of bills
        # date | billno | item | quantity

        column_to_get = 'SUBCATEGORY_DESC'

        bill_data = self.final_data.groupby(['BILLNO', column_to_get]).size()
        print(bill_data.head())
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

        itemsets_dct = self.main("Products.csv", support, itemset_size)

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
        
        
        model = PandasModel(bundled_dataFrame.head(30))
        self.tableViewPredict1 = QTableView(self)
        self.tableViewPredict1.setModel(model)
        self.commonLayout.addWidget(self.tableViewPredict1)
        self.right.setLayout(self.commonLayout)
    
    def main(self,file_location, support, itemset_size):
        candidate_dct = defaultdict(lambda: 0)
        for i in range(itemset_size):
            now = time.time()
            candidate_dct = self.data_pass(file_location, support, pass_nbr=i+1, candidate_dct=candidate_dct)
            print ("pass number %i took %f and found %i candidates" % (i+1, time.time()-now, len(candidate_dct)))
        return candidate_dct
    def data_pass(self,file_location, support, pass_nbr, candidate_dct):
        with open(file_location, 'r') as f:
            for line in f:
                item_lst = line.split(":")     
                candidate_dct = self.update_candidates(item_lst, candidate_dct, pass_nbr)

        candidate_dct = self.clear_items(candidate_dct, support, pass_nbr)

        return candidate_dct
    def update_candidates(self,item_lst, candidate_dct, pass_nbr):
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
    def clear_items(self,candidate_dct, support, pass_nbr):
        for item_tuple, cnt in list(candidate_dct.items()):
            if cnt<support or len(item_tuple)<pass_nbr:
                del candidate_dct[item_tuple]
        return candidate_dct
    def btnstate(self):
        if self.b1.isChecked():
            print("button pressed")
        else:
            print("button released")

    def whichbtn(self,b):
          print("clicked button is "+b.text())
    
    
    def analytics(self):
        for i in reversed(range(self.commonLayout.count())): 
            self.commonLayout.itemAt(i).widget().deleteLater()
        
        box = QHBoxLayout()
        year = QPushButton("Year Summary")
        month = QPushButton("Month Summary")
        week = QPushButton("Week Summary")
        year.clicked.connect(self.yearlyReport)
        month.clicked.connect(self.monthlyReport)
        week.clicked.connect(self.weeklyReport)
        
        w123 = QWidget()
        box.addWidget(year)
        box.addWidget(month)
        box.addWidget(week)
        w123.setLayout(box)
        
        w1234 = QWidget()
        hbox = QHBoxLayout()
        self.tableViewPredictY = QTableView(self)
        self.tableViewPredictM = QTableView(self)
        self.tableViewPredictW = QTableView(self)
        hbox.addWidget(self.tableViewPredictY)
        hbox.addWidget(self.tableViewPredictM)
        hbox.addWidget(self.tableViewPredictW)
        w1234.setLayout(hbox)
        
        parentV = QVBoxLayout()
        parentV.addWidget(w123)
        
        parentV.addWidget(w1234)
        
        w = QWidget()
        w.setLayout(parentV)
        
        self.commonLayout.addWidget(w)
        self.right.setLayout(self.commonLayout)
        
    def yearlyReport(self):
        final_data_new = self.final_data.groupby('SUBCATEGORY_DESC').size()
        final_data_new = final_data_new.reset_index()
        final_data_new.columns = ['Item','Quantity']
        
        model = PandasModel(final_data_new.sort_values('Quantity',ascending=False))
        self.tableViewPredictY.setModel(model)
        
    def monthlyReport(self):
        final_data_new = self.final_data.groupby(['Month','SUBCATEGORY_DESC']).size()
        final_data_new = final_data_new.reset_index()
        final_data_new.columns = ['Month','Item','Quantity']
        
        model = PandasModel(final_data_new.sort_values('Quantity',ascending=False))
        self.tableViewPredictM.setModel(model)
        
    def weeklyReport(self):
        final_data_new = self.final_data.groupby(['SUBCATEGORY_DESC','weekday']).size()
        final_data_new = final_data_new.reset_index()
        final_data_new.columns = ['Item','weekday','Quantity']
        
        model = PandasModel(final_data_new.sort_values('Quantity',ascending=False))
        self.tableViewPredictW.setModel(model)
        
    def bill(self):
        print("Bill function is called")
        # remove other widgets
        try:
            for i in reversed(range(self.commonLayout.count())): 
                self.commonLayout.itemAt(i).widget().deleteLater()
        except :
            print("error")
        # plug them into vertical layout
        
        self.billWidget = QWidget()
        
        layout1 = QHBoxLayout()
        self.manualBillNo = QLineEdit("<Enter Bill No>")
        manualBillNoSubmit = QPushButton("Submit")
        manualBillNoSubmit.clicked.connect(self.manualBillCheck)
        layout1.addWidget(self.manualBillNo)
        layout1.addWidget(manualBillNoSubmit)
        layout1.addStretch()
        layout1.addStretch()
        layout1.addStretch()        
        
        self.layoutBill = QHBoxLayout(self)
        self.combo = QComboBox(self)
        self.combo.activated[str].connect(self.onActivated)
        calButton = QToolButton(self)
        calButton.setIcon(QIcon('./img/cal.png'))
        calButton.setStyleSheet('border: 0px; padding: 0px;')
        calButton.clicked.connect(self.showCalWid)
        self.layoutBill.addWidget(self.combo)
        self.edit = QLineEdit()
        self.layoutBill.addWidget(self.edit)
        self.layoutBill.addWidget(calButton)
        self.layoutBill.addStretch()
        self.layoutBill.addStretch()
        self.layoutBill.addStretch()        
        
        w1 = QWidget()
        w1.setLayout(layout1)
        w2 = QWidget()
        w2.setLayout(self.layoutBill)
        
        finalVerticalLayout = QVBoxLayout()
        finalVerticalLayout.addWidget(w1)
        finalVerticalLayout.addWidget(w2)
        finalVerticalLayout.addStretch()   
        
        self.billWidget.setLayout(finalVerticalLayout)        
        self.commonLayout.addWidget(self.billWidget)
        # just remove bill widget from other classes if exist       
        self.right.setLayout(self.commonLayout)

    def manualBillCheck(self):
        print(self.manualBillNo.text())
        exPopup = ExamplePopup(self.manualBillNo.text(),self.final_data,self)
        exPopup.setGeometry(350, 100, 600, 400)
        exPopup.show()
        
    def onActivated(self, text):      
#         self.lbl.setText(text)
#         self.lbl.adjustSize()  
        print(text)
#         self.bill_data = self.final_data.groupby(['BILLNO', column_to_get]).size()
        exPopup = ExamplePopup(text,self.final_data,self)
        exPopup.setGeometry(350, 100, 600, 400)
        exPopup.show()
        
    def showCalWid(self):
        self.calendar = QCalendarWidget()
        self.calendar.setMinimumDate(QDate(1900, 1, 1))
        self.calendar.setMaximumDate(QDate(3000, 1, 1))
        self.calendar.setGridVisible(True)
        self.calendar.clicked.connect(self.updateDate)
        self.calendar.setWindowFlags(Qt.FramelessWindowHint)
        self.calendar.setStyleSheet('background: white; color: black')
        self.calendar.setGridVisible(True)
        pos = QCursor.pos()
        self.calendar.setGeometry(pos.x(), pos.y(),300, 200)
        self.calendar.show()

    def updateDate(self,calendar):
        getDate = self.calendar.selectedDate().toString()
        from dateutil.parser import parse
        dt = parse(str(getDate))
#         print(dt)
        self.edit.setText(str(dt))
        # delete calender
        self.calendar.deleteLater()
        
        # add combo box
        self.bill_data_temp = self.final_data.groupby(['Date','BILLNO']).size()
        self.bill_data_temp = self.bill_data_temp.reset_index()        
        self.bill_data_temp['Date'] = pd.to_datetime(self.bill_data_temp['Date'], errors='coerce')
        self.bill_data_temp[self.bill_data_temp['Date'] == dt]
        # add vales in combo box
        print(len(self.bill_data_temp['BILLNO'].tolist()))
        for v in self.bill_data_temp['BILLNO'].tolist()[1:2000]:
            self.combo.addItem(v)
    
    
    def __init__(self):
        super().__init__()
        self.title = 'Saamu.ai'
        self.left = 0
        self.top = 0
        self.width = 10000
        self.height = 10000
        self.initUI()
        
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        
        self.bar = self.menuBar()
        self.fileMenu()
        
        # for common usage in right window
        self.commonLayout = QVBoxLayout()
        self.right = QFrame()
        
        self.summaryWidget = QWidget()
        self.billWidget = QWidget()
        # for calender window
        
        self.mainWindow()
        
    def buildExamplePopup(self, item):
            exPopup = ExamplePopup(item.text(), self)
            exPopup.setGeometry(100, 200, 100, 100)
            exPopup.show()

    
class ExamplePopup(QDialog):

    def __init__(self, name,final_data, parent=None):
        super().__init__(parent)
        self.name = name
        self.label = QLabel(self.name, self)   
        print(self.name)
        model = PandasModel(final_data[final_data['BILLNO']==name][['Date','SUBCATEGORY_DESC','BASEPACK_DESC','weekday']])
        self.tableView = QTableView(self)
        self.tableView.setModel(model)        
        self.vboxTable1 = QVBoxLayout()
        self.vboxTable1.addWidget(self.label)
        self.vboxTable1.addWidget(self.tableView)
        self.setLayout(self.vboxTable1)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
       