
# from flask import Flask, render_template,request,jsonify, redirect, session
# from flask_cors import CORS, cross_origin
# from flask_socketio import SocketIO
# import numpy as np
# import pandas as pd
# from datetime import datetime
# import crops
# import pickle
# import calendar
# import random
# #from market_stat import Market
# import croprecomend
# import io
# import os


# from dotenv import load_dotenv
# from config import Config

# from appwrite.client import Client
# from appwrite.services.account import Account
# from appwrite.exception import AppwriteException
# from appwrite.services.storage import Storage


# # Add the production check helper here
# def is_production():
#     return os.environ.get('VERCEL_ENV') == 'production' or os.environ.get('PRODUCTION') == 'true'


# # import matplotlib.pyplot as plt

# app = Flask(__name__)
# app.config.from_object(Config)
# app.config['CORS_HEADERS'] = 'Content-Type'
# app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
# app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
# socketio = SocketIO(app)

# cors = CORS(app, resources={r"/ticker": {"origins": "*"}})

# # commodity_dict = {
# #     "arhar": "static/Arhar.csv",
# #     "bajra": "static/Bajra.csv",
# #     "barley": "static/Barley.csv",
# #     "copra": "static/Copra.csv",
# #     "cotton": "static/Cotton.csv",
# #     "sesamum": "static/Sesamum.csv",
# #     "gram": "static/Gram.csv",
# #     "groundnut": "static/Groundnut.csv",
# #     "jowar": "static/Jowar.csv",
# #     "maize": "static/Maize.csv",
# #     "masoor": "static/Masoor.csv",
# #     "moong": "static/Moong.csv",
# #     "niger": "static/Niger.csv",
# #     "paddy": "static/Paddy.csv",
# #     "ragi": "static/Ragi.csv",
# #     "rape": "static/Rape.csv",
# #     "jute": "static/Jute.csv",
# #     "safflower": "static/Safflower.csv",
# #     "soyabean": "static/Soyabean.csv",
# #     "sugarcane": "static/Sugarcane.csv",
# #     "sunflower": "static/Sunflower.csv",
# #     "urad": "static/Urad.csv",
# #     "wheat": "static/Wheat.csv"
# # }
# commodity_dict = {
#     "arhar": "680220b7000b19e5470b",
#     "bajra": "680220b30004dc8b4590",
#     "barley": "680220ae003249df2282",
#     "copra": "680220a9002d1ffda549",
#     "cotton": "680220a5000a00ef9a0d",
#     "sesamum": "68022042003e073211e4",
#     "gram": "6802208a00037a0a5bfa",
#     "groundnut": "6802208400158728a327",
#     "jowar": "6802207e002276203831",
#     "maize": "680220730024756d13e8",
#     "masoor": "6802206d003937daa9fa",
#     "moong": "680220690003e0ea2c25",
#     "niger": "680220630031664b31be",
#     "paddy": "6802205e0030ada1a89f",
#     "ragi": "6802205900318f8790ff",
#     "rape": "6802205400120d77255a",
#     "jute": "68022078001ce6b8f704",
#     "safflower": "680220480016b66a0099",
#     "soyabean": "6802203c003766174adf",
#     "sugarcane": "68022024003e41ed1eaa",
#     "sunflower": "68021ff80023d86bdb9a",
#     "urad": "68021feb0013dc8f5220",
#     "wheat": "68021fdb00200b466d08"
# }

# annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]
# base = {
#     "Paddy": 1245.5,
#     "Arhar": 3200,
#     "Bajra": 1175,
#     "Barley": 980,
#     "Copra": 5100,
#     "Cotton": 3600,
#     "Sesamum": 4200,
#     "Gram": 2800,
#     "Groundnut": 3700,
#     "Jowar": 1520,
#     "Maize": 1175,
#     "Masoor": 2800,
#     "Moong": 3500,
#     "Niger": 3500,
#     "Ragi": 1500,
#     "Rape": 2500,
#     "Jute": 1675,
#     "Safflower": 2500,
#     "Soyabean": 2200,
#     "Sugarcane": 2250,
#     "Sunflower": 3700,
#     "Urad": 4300,
#     "Wheat": 1350

# }
# commodity_list = []


# # class Commodity:
# #     def __init__(self, csv_name):
# #         self.name = csv_name
# #         dataset = pd.read_csv(csv_name)
# #         self.X = dataset.iloc[:, :-1].values #extracting rows using pandas
# #         self.Y = dataset.iloc[:, 3].values
# #         # Fitting decision tree regression to dataset
# #         from sklearn.tree import DecisionTreeRegressor
# #         depth = random.randrange(7,18) #returns a random depth of the tree from the specified range
# #         self.regressor = DecisionTreeRegressor(max_depth=depth)
# #         self.regressor.fit(self.X, self.Y) #fit() method takes the training data as arguments

# #     def getPredictedValue(self, value):
# #         if value[1]>=2019:
# #             fsa = np.array(value).reshape(1, 3)
# #             #Convert the following 1-D array into a 2-D array.
# #             #The outermost dimension will have 1 array, each with 3 elements.
# #             return self.regressor.predict(fsa)[0]
# #         else:
# #             c=self.X[:,0:2]
# #             x=[]
# #             for i in c:
# #                 x.append(i.tolist())
# #             fsa = [value[0], value[1]]
# #             ind = 0
# #             for i in range(0,len(x)):
# #                 if x[i]==fsa:
# #                     ind=i
# #                     break
# #             return self.Y[i]

# #     def getCropName(self):
# #         a = self.name.split('.')
# #         return a[0]

# # class Commodity:
# #     def __init__(self, file_id):
# #         self.name = file_id
        
# #         # Fetch the file from Appwrite storage
# #         file = storage.get_file(bucket_id=app.config['APPWRITE_BUCKET_ID'], file_id=file_id)
        
# #         # Convert the file's binary data into a pandas dataframe
# #         file_data = file['data']  # Binary content of the file
# #         dataset = pd.read_csv(io.BytesIO(file_data))  # Read CSV from the binary data
        
# #         self.X = dataset.iloc[:, :-1].values
# #         self.Y = dataset.iloc[:, 3].values
        
# #         # Train the decision tree model
# #         from sklearn.tree import DecisionTreeRegressor
# #         depth = random.randrange(7,18)
# #         self.regressor = DecisionTreeRegressor(max_depth=depth)
# #         self.regressor.fit(self.X, self.Y)

# #     def getPredictedValue(self, value):
# #         if value[1] >= 2019:
# #             fsa = np.array(value).reshape(1, 3)
# #             return self.regressor.predict(fsa)[0]
# #         else:
# #             c = self.X[:, 0:2]
# #             x = [i.tolist() for i in c]
# #             fsa = [value[0], value[1]]
# #             ind = 0
# #             for i in range(len(x)):
# #                 if x[i] == fsa:
# #                     ind = i
# #                     break
# #             return self.Y[ind]

# #     def getCropName(self):
# #         a = self.name.split('.')
# #         return a[0]

# class Commodity:
#     def __init__(self, file_id, client):
#         # Reuse the existing client passed as an argument
#         self.storage = Storage(client)

#         self.name = file_id
#         dataset = self.get_csv_from_appwrite(file_id)

#         self.X = dataset.iloc[:, :-1].values
#         self.Y = dataset.iloc[:, 3].values

#         # Fitting decision tree regression to the dataset
#         from sklearn.tree import DecisionTreeRegressor
#         depth = random.randrange(7,18)  # Returns a random depth of the tree from the specified range
#         self.regressor = DecisionTreeRegressor(max_depth=depth)
#         self.regressor.fit(self.X, self.Y)  # fit() method takes the training data as arguments

#     def get_csv_from_appwrite(self, file_id):
#         try:
#             # Fetch the file from Appwrite Storage
#             file = self.storage.get_file(file_id=file_id, bucket_id=app.config['APPWRITE_BUCKET_ID'])
#             file_data = file['data']  # This is the binary content of the file

#             # Convert the binary data into a pandas dataframe
#             dataset = pd.read_csv(io.BytesIO(file_data))  # Read CSV from the binary data
#             return dataset
#         except Exception as e:
#             print(f"Error fetching file from Appwrite: {e}")
#             return None

#     def getPredictedValue(self, value):
#         if value[1] >= 2019:
#             fsa = np.array(value).reshape(1, 3)
#             return self.regressor.predict(fsa)[0]
#         else:
#             c = self.X[:, 0:2]
#             x = [i.tolist() for i in c]
#             fsa = [value[0], value[1]]
#             ind = 0
#             for i in range(len(x)):
#                 if x[i] == fsa:
#                     ind = i
#                     break
#             return self.Y[ind]

#     def getCropName(self):
#         a = self.name.split('.')
#         return a[0]




# @app.route('/')
# def index():
#     if 'email' in session:
#         return render_template('index1.html')
#     return redirect('/login')

# @app.route('/trends')
# def trends():
#     context = {
#         "top3": TopThreeWinners(),
#         "bottom3": TopThreeLosers(),
#         "sixmonths": SixMonthsForecast()
#     }
#     return render_template('trends.html', context=context)

# @app.route('/explore')
# def explore():
#     return render_template('explore.html')

# @app.route('/guide')
# def guide():
#     return render_template('guide.html')

# @app.route('/weather',methods=['GET'])
# def weather():
#     cityname = request.args.get('city')
#     context = {
#         "weatherdesc":  weatherf(cityname)
#     }
#     return render_template('weather.html',context = context)

# def weatherf(cityname):
#     import socket
#     import requests
#     try:
#         send =[]
#         socket.create_connection( ("www.google.com",80) )
#         city = cityname
#         if(cityname == None):
#             pass
#         else:
#             a1 = "http://api.openweathermap.org/data/2.5/forecast?units=metric"
#             a2 = "&q=" + city
#             a3 = "&appid=c6e315d09197cec231495138183954bd"
#             api_address = a1 + a2 + a3
#             res1=requests.get(api_address)
#             data = res1.json()
#             list1 = data['list']
#             res = [sub['main'] for sub in list1]
#             temp = [t['temp'] for t in res]
#             hum = [t['humidity'] for t in res]
#             mini = [t['temp_min'] for t in res]
#             maxim = [t['temp_max'] for t in res]
#             time = [sub['dt_txt'] for sub in list1]
#             degree_sign = u"\N{DEGREE SIGN}" + "C"
#             weather = [sub['weather'] for sub in list1]
#             weatherm,weatherd,icon = [],[],[]
#             for i in range(len(temp)):
#                 for sub in weather[i]:
#                     weatherm.append(sub['main'])
#                     weatherd.append(sub['description'])
#             send.append([time,temp,mini,maxim,hum,weatherm,weatherd])
#     except KeyError as k:
#         print("City Not Found")
#     except OSError as e:
#         print("check network ",e)
#     return send

# @app.route('/chat')
# def sessions():
#     return render_template('session.html')

# # @app.route('/',methods=['GET', 'POST'])
# def messageReceived(methods=['GET', 'POST']):
#     print('message was received!!!')


# @socketio.on('my event')
# def handle_my_custom_event(json, methods=['GET', 'POST']):
#     print('received my event: ' + str(json))
#     socketio.emit('my response', json, callback=messageReceived)


# @app.route('/commodity/<name>')
# def crop_profile(name):
#     # Get the file_id for the commodity from your commodity_dict
#     file_id = commodity_dict.get(name)
    
#     if file_id:
#         # Initialize the Commodity class with the file_id and Appwrite client
#         commodity = Commodity(file_id=file_id, client=client)
        
#         # Use the methods of Commodity class to get the crop's predictions and other data
#         max_crop, min_crop, forecast_crop_values = TwelveMonthsForecast(name)
#         prev_crop_values = TwelveMonthPrevious(name)
#         forecast_x = [i[0] for i in forecast_crop_values]
#         forecast_y = [i[1] for i in forecast_crop_values]
#         previous_x = [i[0] for i in prev_crop_values]
#         previous_y = [i[1] for i in prev_crop_values]
#         current_price = CurrentMonth(name)
        
#         # Get crop data from your crops module (you might want to fetch this dynamically too)
#         crop_data = crops.crop(name)

#         context = {
#             "name": name,
#             "max_crop": max_crop,
#             "min_crop": min_crop,
#             "forecast_values": forecast_crop_values,
#             "forecast_x": str(forecast_x),
#             "forecast_y": forecast_y,
#             "previous_values": prev_crop_values,
#             "previous_x": previous_x,
#             "previous_y": previous_y,
#             "current_price": current_price,
#             "image_url": crop_data[0],
#             "prime_loc": crop_data[1],
#             "type_c": crop_data[2],
#             "export": crop_data[3]
#         }
        
#         # Render the commodity.html template with the context
#         return render_template('commodity.html', context=context)
    
#     else:
#         # If the file_id does not exist for the given commodity, handle the error (e.g., 404 page)
#         return "Commodity data not found", 404


# @app.route('/croprecomd')
# def crop_recommend():
#     return render_template('croprecom.html')

# @app.route('/predict',methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     feature = []
#     for x in request.form.values():
#         feature.append(x)
    
#     pickle_out = open("feature.pickle","wb")
#     pickle.dump(feature, pickle_out)
#     pickle_out.close()
        
#     #final_features = [np.array(int_features)]
#     #prediction = model.predict(final_features)

#     croprecomend.apply()    
#     pickle_in = open("model_output.pickle","rb")
#     output = pickle.load(pickle_in)
#     avail = str(output[0])
#     high = str(output[1])
#     rec = str(output[2])
#     low = str(output[3])
#     day = str(output[4])
#     month = int(output[5])
#     if(month == 1):
#         month = "January."
#     elif(month==2):
#         month = "February."
#     elif(month==3):
#         month = "March."
#     elif(month==4):
#         month = "April."
#     elif(month==5):
#         month = "May."
#     elif(month==6):
#         month = "June."
#     elif(month==7):
#         month = "July."
#     elif(month==8):
#         month = "August."
#     elif(month==9):
#         month = "September."
#     elif(month==10):
#         month = "October."
#     elif(month==11):
#         month = "November."
#     else:
#         month = "December."
    
#     day = day + ', '+ month
#     loc = str(output[6])
#     avail = avail.translate({ord(i): None for i in "{}''"})
#     high = high.translate({ord(i): None for i in "[]''"})
#     rec = rec.translate({ord(i): None for i in "[]''"})
#     low = low.translate({ord(i): None for i in "[]''"})


#     #return render_template('index.html', Avail='AC: {}'.format(output[0]), High ='H: {}'.format(output[1]), avg ='A: {}'.format(output[2], low ='N: {}'.format(output[3])))
#     return render_template('croprecom.html', Avail=avail, Low=low, High=high, Rec=rec,Day = day,Loc = loc)
    
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = croprecomend.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

# @app.route('/ticker/<item>/<number>')
# @cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
# def ticker(item, number):
#     n = int(number)
#     i = int(item)
#     data = SixMonthsForecast()
#     context = str(data[n][i])

#     if i == 2 or i == 5:
#         context = '₹' + context
#     elif i == 3 or i == 6:
#         context = context + '%'
#     return context


# def TopThreeWinners():
#     current_month = datetime.now().month
#     current_year = datetime.now().year
#     current_rainfall = annual_rainfall[current_month - 1]
#     prev_month = current_month - 1
#     prev_rainfall = annual_rainfall[prev_month - 1]
#     current_month_prediction = []
#     prev_month_prediction = []
#     change = []

#     for i in commodity_list:
#         current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
#         current_month_prediction.append(current_predict)
#         prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
#         prev_month_prediction.append(prev_predict)
#         change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))
#     sorted_change = change
#     sorted_change.sort(reverse=True)
#     # print(sorted_change)
#     to_send = []
#     for j in range(0, 3):
#         perc, i = sorted_change[j]
#         name = commodity_list[i].getCropName().split('/')[1]
#         to_send.append([name, round((current_month_prediction[i] * base[name]) / 100, 2), round(perc, 2)])
#     #print(to_send)
#     return to_send


# def TopThreeLosers():
#     current_month = datetime.now().month
#     current_year = datetime.now().year
#     current_rainfall = annual_rainfall[current_month - 1]
#     prev_month = current_month - 1
#     prev_rainfall = annual_rainfall[prev_month - 1]
#     current_month_prediction = []
#     prev_month_prediction = []
#     change = []

#     for i in commodity_list:
#         current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
#         current_month_prediction.append(current_predict)
#         prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
#         prev_month_prediction.append(prev_predict)
#         change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))
#     sorted_change = change
#     sorted_change.sort()
#     to_send = []
#     for j in range(0, 3):
#         perc, i = sorted_change[j]
#         name = commodity_list[i].getCropName().split('/')[1]
#         to_send.append([name, round((current_month_prediction[i] * base[name]) / 100, 2), round(perc, 2)])
#    # print(to_send)
#     return to_send


# def SixMonthsForecast():
#     month1=[]
#     month2=[]
#     month3=[]
#     month4=[]
#     month5=[]
#     month6=[]
#     for i in commodity_list:
#         crop=SixMonthsForecastHelper(i.getCropName())
#         k=0
#         for j in crop:
#             time = j[0]
#             price = j[1]
#             change = j[2]
#             if k==0:
#                 month1.append((price,change,i.getCropName().split("/")[1],time))
#             elif k==1:
#                 month2.append((price,change,i.getCropName().split("/")[1],time))
#             elif k==2:
#                 month3.append((price,change,i.getCropName().split("/")[1],time))
#             elif k==3:
#                 month4.append((price,change,i.getCropName().split("/")[1],time))
#             elif k==4:
#                 month5.append((price,change,i.getCropName().split("/")[1],time))
#             elif k==5:
#                 month6.append((price,change,i.getCropName().split("/")[1],time))
#             k+=1
#     month1.sort()
#     month2.sort()
#     month3.sort()
#     month4.sort()
#     month5.sort()
#     month6.sort()
#     crop_month_wise=[]
#     crop_month_wise.append([month1[0][3],month1[len(month1)-1][2],month1[len(month1)-1][0],month1[len(month1)-1][1],month1[0][2],month1[0][0],month1[0][1]])
#     crop_month_wise.append([month2[0][3],month2[len(month2)-1][2],month2[len(month2)-1][0],month2[len(month2)-1][1],month2[0][2],month2[0][0],month2[0][1]])
#     crop_month_wise.append([month3[0][3],month3[len(month3)-1][2],month3[len(month3)-1][0],month3[len(month3)-1][1],month3[0][2],month3[0][0],month3[0][1]])
#     crop_month_wise.append([month4[0][3],month4[len(month4)-1][2],month4[len(month4)-1][0],month4[len(month4)-1][1],month4[0][2],month4[0][0],month4[0][1]])
#     crop_month_wise.append([month5[0][3],month5[len(month5)-1][2],month5[len(month5)-1][0],month5[len(month5)-1][1],month5[0][2],month5[0][0],month5[0][1]])
#     crop_month_wise.append([month6[0][3],month6[len(month6)-1][2],month6[len(month6)-1][0],month6[len(month6)-1][1],month6[0][2],month6[0][0],month6[0][1]])

#    # print(crop_month_wise)
#     return crop_month_wise

# def SixMonthsForecastHelper(name):
#     current_month = datetime.now().month
#     current_year = datetime.now().year
#     current_rainfall = annual_rainfall[current_month - 1]
#     name = name.split("/")[1]
#     name = name.lower()
#     commodity = commodity_list[0]
#     for i in commodity_list:
#         if name == str(i):
#             commodity = i
#             break
#     month_with_year = []
#     for i in range(1, 7):
#         if current_month + i <= 12:
#             month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
#         else:
#             month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[current_month + i - 13]))
#     wpis = []
#     current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
#     change = []

#     for m, y, r in month_with_year:
#         current_predict = commodity.getPredictedValue([float(m), y, r])
#         wpis.append(current_predict)
#         change.append(((current_predict - current_wpi) * 100) / current_wpi)

#     crop_price = []
#     for i in range(0, len(wpis)):
#         m, y, r = month_with_year[i]
#         x = datetime(y, m, 1)
#         x = x.strftime("%b %y")
#         crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2) , round(change[i], 2)])

#    # print("Crop_Price: ", crop_price)
#     return crop_price

# def CurrentMonth(name):
#     current_month = datetime.now().month
#     current_year = datetime.now().year
#     current_rainfall = annual_rainfall[current_month - 1]
#     name = name.lower()
#     commodity = commodity_list[0]
#     for i in commodity_list:
#         if name == str(i):
#             commodity = i
#             break
#     current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
#     current_price = (base[name.capitalize()]*current_wpi)/100
#     return current_price

# def TwelveMonthsForecast(name):
#     current_month = datetime.now().month
#     current_year = datetime.now().year
#     current_rainfall = annual_rainfall[current_month - 1]
#     name = name.lower()
#     commodity = commodity_list[0]
#     for i in commodity_list:
#         if name == str(i):
#             commodity = i
#             break
#     month_with_year = []
#     for i in range(1, 13):
#         if current_month + i <= 12:
#             month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
#         else:
#             month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[current_month + i - 13]))
#     max_index = 0
#     min_index = 0
#     max_value = 0
#     min_value = 9999
#     wpis = []
#     current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
#     change = []

#     for m, y, r in month_with_year:
#         current_predict = commodity.getPredictedValue([float(m), y, r])
#         if current_predict > max_value:
#             max_value = current_predict
#             max_index = month_with_year.index((m, y, r))
#         if current_predict < min_value:
#             min_value = current_predict
#             min_index = month_with_year.index((m, y, r))
#         wpis.append(current_predict)
#         change.append(((current_predict - current_wpi) * 100) / current_wpi)

#     max_month, max_year, r1 = month_with_year[max_index]
#     min_month, min_year, r2 = month_with_year[min_index]
#     min_value = min_value * base[name.capitalize()] / 100
#     max_value = max_value * base[name.capitalize()] / 100
#     crop_price = []
#     for i in range(0, len(wpis)):
#         m, y, r = month_with_year[i]
#         x = datetime(y, m, 1)
#         x = x.strftime("%b %y")
#         crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2) , round(change[i], 2)])
#    # print("forecasr", wpis)
#     x = datetime(max_year,max_month,1)
#     x = x.strftime("%b %y")
#     max_crop = [x, round(max_value,2)]
#     x = datetime(min_year, min_month, 1)
#     x = x.strftime("%b %y")
#     min_crop = [x, round(min_value,2)]

#     return max_crop, min_crop, crop_price


# def TwelveMonthPrevious(name):
#     name = name.lower()
#     current_month = datetime.now().month
#     current_year = datetime.now().year
#     current_rainfall = annual_rainfall[current_month - 1]
#     commodity = commodity_list[0]
#     wpis = []
#     crop_price = []
#     for i in commodity_list:
#         if name == str(i):
#             commodity = i
#             break
#     month_with_year = []
#     for i in range(1, 13):
#         if current_month - i >= 1:
#             month_with_year.append((current_month - i, current_year, annual_rainfall[current_month - i - 1]))
#         else:
#             month_with_year.append((current_month - i + 12, current_year - 1, annual_rainfall[current_month - i + 11]))

#     for m, y, r in month_with_year:
#         current_predict = commodity.getPredictedValue([float(m), 2013, r])
#         wpis.append(current_predict)

#     for i in range(0, len(wpis)):
#         m, y, r = month_with_year[i]
#         x = datetime(y,m,1)
#         x = x.strftime("%b %y")
#         crop_price.append([x, round((wpis[i]* base[name.capitalize()]) / 100, 2)])
#    # print("previous ", wpis)
#     new_crop_price =[]
#     for i in range(len(crop_price)-1,-1,-1):
#         new_crop_price.append(crop_price[i])
#     return new_crop_price

# @app.route('/fertilizer_info',methods=['POST','GET'])
# def fertilizer_info():
#     data = pd.read_csv('static/final_fertilizer.csv')
#     crops = data['Crop'].unique()

#     if request.method == 'GET':
#         crop_se = request.args.get('manager')
#         query = data[data['Crop']==crop_se]
#         query = query['query'].unique()
#         queryArr = []
#         if len(query):
#             for query_name in query:
#                 queryObj = {}
#                 queryObj['name'] = query_name
#                 print(query_name)
#                 queryArr.append(queryObj)
#             return jsonify({'data':render_template('fertilizer.html',crops=crops,crop_len=len(crops)),'query':queryArr})

#     if request.method == 'POST':
#         crop_name = request.form['crop']
#         query_type = request.form['query']
#         query = data[data['Crop']==crop_name]
#         answer = query[query['query']== query_type]
#         answer = answer['KCCAns'].unique()
#         protection = []
#         for index in answer:
#             protection.append(index)

#         return render_template('fertilizer.html',protection=protection,protection_len=len(protection),display=True,crops=crops,crop_len=len(crops))


#     return render_template('fertilizer.html',crops=crops,crop_len=len(crops),query_len=0)



# @app.route('/shop',methods=['POST','GET'])
# def shop():
#     if request.method == 'POST':
#         city = request.form['city']
#         print(city)

#         return render_template('fertilizer_shop.html',city=city,data=True)

#     return render_template('fertilizer_shop.html')


# # Appwrite Configuration

# load_dotenv()

# # Initialize Appwrite Client
# client = Client()
# (client
#     .set_endpoint(app.config['APPWRITE_ENDPOINT'])  # e.g., "https://cloud.appwrite.io/v1"
#     .set_project(app.config['APPWRITE_PROJECT_ID'])  # Your Appwrite project ID
# )

# account = Account(client)
# storage = Storage(client)

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']
#         try:
#             account.create(user_id='unique()', email=email, password=password)
#             return redirect('/login')
#         except AppwriteException as e:
#             return f"Signup Error: {e.message}"
#     return render_template('signup.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form['email']
#         password = request.form['password']
#         try:
#             session_response = account.create_email_password_session(email, password)
#             session['email'] = email
#             # return f"Logged in as {email} <br><a href='/logout'>Logout</a>"
#             return render_template('index1.html', email=email)

#         except AppwriteException as e:
#             return f"Login Error: {e.message}"
#     return render_template('login.html')


# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect('/login')

# def load_csv_from_appwrite(file_id):
#     try:
#         file_bytes = storage.get_file_download(app.config['APPWRITE_BUCKET_ID'], file_id)
#         csv_content = io.BytesIO(file_bytes)
#         df = pd.read_csv(csv_content)
#         return df
#     except Exception as e:
#         print(f"Error loading CSV: {e}")
#         return None



# # if __name__ == "__main__":
# #     arhar = Commodity(commodity_dict["arhar"])
# #     commodity_list.append(arhar)
# #     bajra = Commodity(commodity_dict["bajra"])
# #     commodity_list.append(bajra)
# #     barley = Commodity(commodity_dict["barley"])
# #     commodity_list.append(barley)
# #     copra = Commodity(commodity_dict["copra"])
# #     commodity_list.append(copra)
# #     cotton = Commodity(commodity_dict["cotton"])
# #     commodity_list.append(cotton)
# #     sesamum = Commodity(commodity_dict["sesamum"])
# #     commodity_list.append(sesamum)
# #     gram = Commodity(commodity_dict["gram"])
# #     commodity_list.append(gram)
# #     groundnut = Commodity(commodity_dict["groundnut"])
# #     commodity_list.append(groundnut)
# #     jowar = Commodity(commodity_dict["jowar"])
# #     commodity_list.append(jowar)
# #     maize = Commodity(commodity_dict["maize"])
# #     commodity_list.append(maize)
# #     masoor = Commodity(commodity_dict["masoor"])
# #     commodity_list.append(masoor)
# #     moong = Commodity(commodity_dict["moong"])
# #     commodity_list.append(moong)
# #     niger = Commodity(commodity_dict["niger"])
# #     commodity_list.append(niger)
# #     paddy = Commodity(commodity_dict["paddy"])
# #     commodity_list.append(paddy)
# #     ragi = Commodity(commodity_dict["ragi"])
# #     commodity_list.append(ragi)
# #     rape = Commodity(commodity_dict["rape"])
# #     commodity_list.append(rape)
# #     jute = Commodity(commodity_dict["jute"])
# #     commodity_list.append(jute)
# #     safflower = Commodity(commodity_dict["safflower"])
# #     commodity_list.append(safflower)
# #     soyabean = Commodity(commodity_dict["soyabean"])
# #     commodity_list.append(soyabean)
# #     sugarcane = Commodity(commodity_dict["sugarcane"])
# #     commodity_list.append(sugarcane)
# #     sunflower = Commodity(commodity_dict["sunflower"])
# #     commodity_list.append(sunflower)
# #     urad = Commodity(commodity_dict["urad"])
# #     commodity_list.append(urad)
# #     wheat = Commodity(commodity_dict["wheat"])
# #     commodity_list.append(wheat)

# #     socketio.run(app)

# # if __name__ == "__main__":
# #     # Initialize the commodity list
# #     commodity_list = []

# #     # Replace the CSV file paths with Appwrite file IDs
# #     for commodity_name, file_id in commodity_dict.items():
# #         # Fetch the data from Appwrite storage using file ID
# #         df = Commodity.get_csv_from_appwrite(file_id)  # Assuming this function is implemented to fetch data from Appwrite
        
# #         # Create a Commodity object using the data fetched from Appwrite (a dataframe in this case)
# #         commodity = Commodity(file_id, client)  # Passing file_id and client to the Commodity constructor
# #         commodity_list.append(commodity)

# #     # Run your application
# #     socketio.run(app)

# if __name__ == "__main__":
#     commodity_list = []

#     for commodity_name, file_id in commodity_dict.items():
#         try:
#             commodity = Commodity(file_id, client)
#             commodity_list.append(commodity)
#         except Exception as e:
#             print(f"Error initializing commodity {commodity_name}: {e}")

# if __name__ == "__main__":
#     # For local development
#     if os.environ.get('VERCEL_ENV') != 'production':
#         socketio.run(app)
#     # For Vercel or other cloud platforms
#     else:
#         # Vercel will use WSGI entry point defined in vercel.json
#         pass

