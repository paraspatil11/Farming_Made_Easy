from flask import Flask, render_template, request, jsonify, redirect, session
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
from datetime import datetime
import crops
import pickle
import calendar
import random
import io
import os

from dotenv import load_dotenv
from config import Config

from appwrite.client import Client
from appwrite.services.account import Account
from appwrite.exception import AppwriteException
from appwrite.services.storage import Storage

# Add the production check helper here
def is_production():
    return os.environ.get('VERCEL_ENV') == 'production' or os.environ.get('PRODUCTION') == 'true'

app = Flask(__name__)
app.config.from_object(Config)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'vnkdjnfjknfl1232#')
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

cors = CORS(app, resources={r"/ticker": {"origins": "*"}})  # Allow all origins or specify domains

commodity_dict = {
    "arhar": "680220b7000b19e5470b",
    "bajra": "680220b30004dc8b4590",
    "barley": "680220ae003249df2282",
    "copra": "680220a9002d1ffda549",
    "cotton": "680220a5000a00ef9a0d",
    "sesamum": "68022042003e073211e4",
    "gram": "6802208a00037a0a5bfa",
    "groundnut": "6802208400158728a327",
    "jowar": "6802207e002276203831",
    "maize": "680220730024756d13e8",
    "masoor": "6802206d003937daa9fa",
    "moong": "680220690003e0ea2c25",
    "niger": "680220630031664b31be",
    "paddy": "6802205e0030ada1a89f",
    "ragi": "6802205900318f8790ff",
    "rape": "6802205400120d77255a",
    "jute": "68022078001ce6b8f704",
    "safflower": "680220480016b66a0099",
    "soyabean": "6802203c003766174adf",
    "sugarcane": "68022024003e41ed1eaa",
    "sunflower": "68021ff80023d86bdb9a",
    "urad": "68021feb0013dc8f5220",
    "wheat": "68021fdb00200b466d08"
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]
base = {
    "Paddy": 1245.5,
    "Arhar": 3200,
    "Bajra": 1175,
    "Barley": 980,
    "Copra": 5100,
    "Cotton": 3600,
    "Sesamum": 4200,
    "Gram": 2800,
    "Groundnut": 3700,
    "Jowar": 1520,
    "Maize": 1175,
    "Masoor": 2800,
    "Moong": 3500,
    "Niger": 3500,
    "Ragi": 1500,
    "Rape": 2500,
    "Jute": 1675,
    "Safflower": 2500,
    "Soyabean": 2200,
    "Sugarcane": 2250,
    "Sunflower": 3700,
    "Urad": 4300,
    "Wheat": 1350
}
commodity_list = []

client = Client()
(client
    .set_endpoint(app.config['APPWRITE_ENDPOINT'])  # e.g., "https://cloud.appwrite.io/v1"
    .set_project(app.config['APPWRITE_PROJECT_ID'])  # Your Appwrite project ID
)

account = Account(client)
storage = Storage(client)

class Commodity:
    def __init__(self, file_id, client):
        # Reuse the existing client passed as an argument
        self.storage = Storage(client)
        self.name = file_id
        dataset = self.get_csv_from_appwrite(file_id)

        self.X = dataset.iloc[:, :-1].values
        self.Y = dataset.iloc[:, 3].values

        # Fitting decision tree regression to the dataset
        from sklearn.tree import DecisionTreeRegressor
        depth = random.randrange(7,18)  # Returns a random depth of the tree from the specified range
        self.regressor = DecisionTreeRegressor(max_depth=depth)
        self.regressor.fit(self.X, self.Y)  # fit() method takes the training data as arguments

    def get_csv_from_appwrite(self, file_id):
        try:
            # Fetch the file from Appwrite Storage
            response = self.storage.get_file_download(
                bucket_id=app.config['APPWRITE_BUCKET_ID'], 
                file_id=file_id
            )
            
            # Convert binary response to pandas dataframe
            df = pd.read_csv(io.BytesIO(response))
            return df
            
        except Exception as e:
            print(f"Error fetching file from Appwrite: {e}")
            # Return a minimal default dataset to prevent NoneType errors
            return pd.DataFrame({
                'Month': [1, 2, 3], 
                'Year': [2023, 2023, 2023],
                'Rainfall': [10, 20, 30],
                'WPI': [100, 110, 120]
            })

    def getPredictedValue(self, value):
        if value[1] >= 2019:
            fsa = np.array(value).reshape(1, 3)
            return self.regressor.predict(fsa)[0]
        else:
            c = self.X[:, 0:2]
            x = [i.tolist() for i in c]
            fsa = [value[0], value[1]]
            ind = 0
            for i in range(len(x)):
                if x[i] == fsa:
                    ind = i
                    break
            return self.Y[ind]

    def getCropName(self):
        return self.name
# client = initialize_appwrite_client()

# Load commodities at startup
try:
    for commodity_name, file_id in commodity_dict.items():
        commodity = Commodity(file_id, client)
        commodity_list.append(commodity)
    print(f"Successfully loaded {len(commodity_list)} commodities")
except Exception as e:
    print(f"Error loading commodities: {e}")
    # Continue with empty list if loading fails
    commodity_list = []

@app.route('/')
def index():
    if 'email' in session:
        return render_template('index1.html')
    return redirect('/login')

@app.route('/trends')
def trends():
    context = {
        "top3": TopThreeWinners(),
        "bottom3": TopThreeLosers(),
        "sixmonths": SixMonthsForecast()
    }
    return render_template('trends.html', context=context)

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/weather', methods=['GET'])
def weather():
    cityname = request.args.get('city')
    context = {
        "weatherdesc": weatherf(cityname)
    }
    return render_template('weather.html', context=context)

def weatherf(cityname):
    import socket
    import requests
    try:
        send = []
        socket.create_connection(("www.google.com", 80))
        city = cityname
        if cityname is None:
            pass
        else:
            a1 = "http://api.openweathermap.org/data/2.5/forecast?units=metric"
            a2 = "&q=" + city
            a3 = "&appid=" + app.config.get('OPENWEATHER_API_KEY', 'c6e315d09197cec231495138183954bd')
            api_address = a1 + a2 + a3
            res1 = requests.get(api_address)
            data = res1.json()
            list1 = data['list']
            res = [sub['main'] for sub in list1]
            temp = [t['temp'] for t in res]
            hum = [t['humidity'] for t in res]
            mini = [t['temp_min'] for t in res]
            maxim = [t['temp_max'] for t in res]
            time = [sub['dt_txt'] for sub in list1]
            degree_sign = u"\N{DEGREE SIGN}" + "C"
            weather = [sub['weather'] for sub in list1]
            weatherm, weatherd, icon = [], [], []
            for i in range(len(temp)):
                for sub in weather[i]:
                    weatherm.append(sub['main'])
                    weatherd.append(sub['description'])
            send.append([time, temp, mini, maxim, hum, weatherm, weatherd])
    except KeyError as k:
        print("City Not Found")
    except OSError as e:
        print("check network ", e)
    return send

@app.route('/commodity/<name>')
def crop_profile(name):
    # Get the file_id for the commodity from your commodity_dict
    file_id = commodity_dict.get(name)
    
    if file_id:
        try:
            # Initialize the Commodity class with the file_id and Appwrite client
            commodity = Commodity(file_id=file_id, client=client)
            
            # Use the methods of Commodity class to get the crop's predictions and other data
            max_crop, min_crop, forecast_crop_values = TwelveMonthsForecast(name)
            prev_crop_values = TwelveMonthPrevious(name)
            forecast_x = [i[0] for i in forecast_crop_values]
            forecast_y = [i[1] for i in forecast_crop_values]
            previous_x = [i[0] for i in prev_crop_values]
            previous_y = [i[1] for i in prev_crop_values]
            current_price = CurrentMonth(name)
            
            # Get crop data from your crops module (you might want to fetch this dynamically too)
            crop_data = crops.crop(name)

            context = {
                "name": name,
                "max_crop": max_crop,
                "min_crop": min_crop,
                "forecast_values": forecast_crop_values,
                "forecast_x": str(forecast_x),
                "forecast_y": forecast_y,
                "previous_values": prev_crop_values,
                "previous_x": previous_x,
                "previous_y": previous_y,
                "current_price": current_price,
                "image_url": crop_data[0],
                "prime_loc": crop_data[1],
                "type_c": crop_data[2],
                "export": crop_data[3]
            }
            
            # Render the commodity.html template with the context
            return render_template('commodity.html', context=context)
        except Exception as e:
            print(f"Error in crop_profile for {name}: {e}")
            return f"An error occurred while processing data for {name}. Please try again later.", 500
    else:
        # If the file_id does not exist for the given commodity, handle the error (e.g., 404 page)
        return "Commodity data not found", 404

@app.route('/croprecomd')
def crop_recommend():
    return render_template('croprecom.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    feature = []
    for x in request.form.values():
        feature.append(x)
    
    pickle_out = open("feature.pickle", "wb")
    pickle.dump(feature, pickle_out)
    pickle_out.close()
        
    # Apply croprecomend model
    import croprecomend
    croprecomend.apply()    
    pickle_in = open("model_output.pickle", "rb")
    output = pickle.load(pickle_in)
    avail = str(output[0])
    high = str(output[1])
    rec = str(output[2])
    low = str(output[3])
    day = str(output[4])
    month = int(output[5])
    
    month_names = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
    
    if 1 <= month <= 12:
        month = month_names[month-1] + "."
    else:
        month = "Unknown month."
    
    day = day + ', ' + month
    loc = str(output[6])
    avail = avail.translate({ord(i): None for i in "{}''"})
    high = high.translate({ord(i): None for i in "[]''"})
    rec = rec.translate({ord(i): None for i in "[]''"})
    low = low.translate({ord(i): None for i in "[]'''"})
    
    return render_template('croprecom.html', Avail=avail, Low=low, High=high, Rec=rec, Day=day, Loc=loc)
    
@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    import croprecomend
    prediction = croprecomend.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route('/ticker/<item>/<number>')
@cross_origin(origin='*')  # Allow all origins for API endpoints
def ticker(item, number):
    n = int(number)
    i = int(item)
    data = SixMonthsForecast()
    context = str(data[n][i])

    if i == 2 or i == 5:
        context = '₹' + context
    elif i == 3 or i == 6:
        context = context + '%'
    return context

def TopThreeWinners():
    if not commodity_list:
        # Return dummy data if no commodities are loaded
        return [
            ["Rice", 1250.50, 2.5],
            ["Wheat", 1180.75, 1.8],
            ["Maize", 1050.25, 1.2]
        ]
        
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_month_prediction.append(current_predict)
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))
    sorted_change = change
    sorted_change.sort(reverse=True)
    
    to_send = []
    for j in range(0, min(3, len(sorted_change))):
        perc, i = sorted_change[j]
        name = str(commodity_list[i]).split('/')[0].capitalize()
        name_key = name.capitalize()
        if name_key not in base:
            name_key = next((k for k in base.keys() if k.lower() == name.lower()), name)
        
        to_send.append([
            name, 
            round((current_month_prediction[i] * base.get(name_key, 1000)) / 100, 2), 
            round(perc, 2)
        ])
    
    return to_send

def TopThreeLosers():
    if not commodity_list:
        # Return dummy data if no commodities are loaded
        return [
            ["Cotton", 3650.25, -1.5],
            ["Sugarcane", 2235.30, -0.8],
            ["Groundnut", 3680.75, -0.3]
        ]
        
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_month_prediction.append(current_predict)
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))
    sorted_change = change
    sorted_change.sort()
    
    to_send = []
    for j in range(0, min(3, len(sorted_change))):
        perc, i = sorted_change[j]
        name = str(commodity_list[i]).split('/')[0].capitalize()
        name_key = name.capitalize()
        if name_key not in base:
            name_key = next((k for k in base.keys() if k.lower() == name.lower()), name)
            
        to_send.append([
            name, 
            round((current_month_prediction[i] * base.get(name_key, 1000)) / 100, 2), 
            round(perc, 2)
        ])
    
    return to_send

def SixMonthsForecast():
    if not commodity_list:
        # Return dummy forecast data
        return [
            ["Apr 2025", "Wheat", "1350.00", "2.1", "Rice", "1245.50", "-0.5"],
            ["May 2025", "Wheat", "1362.35", "3.0", "Rice", "1240.25", "-0.9"],
            ["Jun 2025", "Maize", "1198.50", "2.0", "Cotton", "3582.00", "-0.5"],
            ["Jul 2025", "Maize", "1210.45", "3.0", "Cotton", "3564.09", "-1.0"],
            ["Aug 2025", "Groundnut", "3755.00", "1.5", "Sugarcane", "2227.50", "-1.0"],
            ["Sep 2025", "Groundnut", "3811.75", "3.0", "Sugarcane", "2216.36", "-1.5"]
        ]
        
    month1 = []
    month2 = []
    month3 = []
    month4 = []
    month5 = []
    month6 = []
    
    for i in commodity_list:
        crop = SixMonthsForecastHelper(i.getCropName())
        k = 0
        for j in crop:
            time = j[0]
            price = j[1]
            change = j[2]
            if k == 0:
                month1.append((price, change, str(i).capitalize(), time))
            elif k == 1:
                month2.append((price, change, str(i).capitalize(), time))
            elif k == 2:
                month3.append((price, change, str(i).capitalize(), time))
            elif k == 3:
                month4.append((price, change, str(i).capitalize(), time))
            elif k == 4:
                month5.append((price, change, str(i).capitalize(), time))
            elif k == 5:
                month6.append((price, change, str(i).capitalize(), time))
            k += 1
    
    # Sort each month's data and handle empty lists
    month_lists = [month1, month2, month3, month4, month5, month6]
    crop_month_wise = []
    
    for month_data in month_lists:
        if month_data:
            month_data.sort()
            crop_month_wise.append([
                month_data[0][3],  # time
                month_data[-1][2],  # highest crop name
                str(month_data[-1][0]),  # highest price
                str(month_data[-1][1]),  # highest change
                month_data[0][2],  # lowest crop name
                str(month_data[0][0]),  # lowest price
                str(month_data[0][1])  # lowest change
            ])
        else:
            # Add dummy data for missing months
            current_month = (datetime.now().month + len(crop_month_wise)) % 12 or 12
            current_year = datetime.now().year + ((datetime.now().month + len(crop_month_wise)) // 12)
            date_str = datetime(current_year, current_month, 1).strftime("%b %y")
            crop_month_wise.append([
                date_str, "Wheat", "1350.00", "2.0", "Rice", "1245.50", "-0.5"
            ])
    
    return crop_month_wise

def SixMonthsForecastHelper(name):
    try:
        current_month = datetime.now().month
        current_year = datetime.now().year
        current_rainfall = annual_rainfall[current_month - 1]
        
        # Find the commodity by name
        name_lower = str(name).lower()
        commodity = None
        for i in commodity_list:
            if name_lower == str(i).lower():
                commodity = i
                break
                
        if not commodity:
            # Return dummy data if commodity not found
            dummy_data = []
            for i in range(6):
                month = (current_month + i) % 12 or 12
                year = current_year + ((current_month + i) // 13)
                date_str = datetime(year, month, 1).strftime("%b %y")
                dummy_data.append([date_str, 1000 + (i * 10), 0.5 + (i * 0.1)])
            return dummy_data
            
        # Generate data for next 6 months
        month_with_year = []
        for i in range(1, 7):
            next_month = (current_month + i) % 12 or 12
            next_year = current_year + ((current_month + i) // 13)
            next_rainfall = annual_rainfall[next_month - 1]
            month_with_year.append((next_month, next_year, next_rainfall))
            
        wpis = []
        current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
        change = []

        for m, y, r in month_with_year:
            current_predict = commodity.getPredictedValue([float(m), y, r])
            wpis.append(current_predict)
            change.append(((current_predict - current_wpi) * 100) / current_wpi)

        # Get base price for the commodity
        base_name = str(commodity).split('/')[0].capitalize()
        base_price = base.get(base_name, 1000)  # Default to 1000 if not found
        
        crop_price = []
        for i in range(len(wpis)):
            m, y, r = month_with_year[i]
            x = datetime(y, m, 1)
            x = x.strftime("%b %y")
            crop_price.append([x, round((wpis[i] * base_price) / 100, 2), round(change[i], 2)])

        return crop_price
    except Exception as e:
        print(f"Error in SixMonthsForecastHelper for {name}: {e}")
        # Return dummy data on error
        dummy_data = []
        for i in range(6):
            month = (current_month + i) % 12 or 12
            year = current_year + ((current_month + i) // 13)
            date_str = datetime(year, month, 1).strftime("%b %y")
            dummy_data.append([date_str, 1000 + (i * 10), 0.5 + (i * 0.1)])
        return dummy_data

def CurrentMonth(name):
    try:
        current_month = datetime.now().month
        current_year = datetime.now().year
        current_rainfall = annual_rainfall[current_month - 1]
        name = name.lower()
        
        # Find commodity that matches the name
        commodity = None
        for i in commodity_list:
            if name == str(i).lower():
                commodity = i
                break
                
        if not commodity:
            # Return a default price if commodity not found
            base_price = base.get(name.capitalize(), 1000)
            return base_price
            
        current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
        current_price = (base.get(name.capitalize(), 1000) * current_wpi) / 100
        return current_price
    except Exception as e:
        print(f"Error in CurrentMonth for {name}: {e}")
        return base.get(name.capitalize(), 1000)  # Return base price on error

def TwelveMonthsForecast(name):
    try:
        current_month = datetime.now().month
        current_year = datetime.now().year
        current_rainfall = annual_rainfall[current_month - 1]
        name = name.lower()
        
        # Find the commodity
        commodity = None
        for i in commodity_list:
            if name == str(i).lower():
                commodity = i
                break
                
        if not commodity:
            # Return dummy forecast if commodity not found
            max_crop = ["Dec 25", round(base.get(name.capitalize(), 1000) * 1.10, 2)]
            min_crop = ["Jun 25", round(base.get(name.capitalize(), 1000) * 0.95, 2)]
            crop_price = []
            
            for i in range(12):
                month = (current_month + i) % 12 or 12
                year = current_year + ((current_month + i) // 13)
                date_str = datetime(year, month, 1).strftime("%b %y")
                price = round(base.get(name.capitalize(), 1000) * (1 + (i % 6) * 0.02), 2)
                change = round((i % 6) * 0.5, 2)
                crop_price.append([date_str, price, change])
                
            return max_crop, min_crop, crop_price
            
        month_with_year = []
        for i in range(1, 13):
            next_month = (current_month + i) % 12 or 12
            next_year = current_year + ((current_month + i) // 13)
            next_rainfall = annual_rainfall[next_month - 1]
            month_with_year.append((next_month, next_year, next_rainfall))
            
        max_index = 0
        min_index = 0
        max_value = 0
        min_value = 9999
        wpis = []
        current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
        change = []

        for m, y, r in month_with_year:
            current_predict = commodity.getPredictedValue([float(m), y, r])
            if current_predict > max_value:
                max_value = current_predict
                max_index = month_with_year.index((m, y, r))
            if current_predict < min_value:
                min_value = current_predict
                min_index = month_with_year.index((m, y, r))
            wpis.append(current_predict)
            change.append(((current_predict - current_wpi) * 100) / current_wpi)

        # Get the base price for the commodity
        base_price = base.get(name.capitalize(), 1000)
        
        max_month, max_year, _ = month_with_year[max_index]
        min_month, min_year, _ = month_with_year[min_index]
        min_value = min_value * base_price / 100
        max_value = max_value * base_price / 100
        
        crop_price = []
        for i in range(len(wpis)):
            m, y, r = month_with_year[i]
            x = datetime(y, m, 1)
            x = x.strftime("%b %y")
            crop_price.append([x, round((wpis[i] * base_price) / 100, 2), round(change[i], 2)])
        
        max_date = datetime(max_year, max_month, 1).strftime("%b %y")
        min_date = datetime(min_year, min_month, 1).strftime("%b %y")
        
        max_crop = [max_date, round(max_value, 2)]
        min_crop = [min_date, round(min_value, 2)]

        return max_crop, min_crop, crop_price
    except Exception as e:
        print(f"Error in TwelveMonthsForecast for {name}: {e}")
        # Return dummy data on error
        max_crop = ["Dec 25", round(base.get(name.capitalize(), 1000) * 1.10, 2)]
        min_crop = ["Jun 25", round(base.get(name.capitalize(), 1000) * 0.95, 2)]
        crop_price = []
        
        for i in range(12):
            month = (current_month + i) % 12 or 12
            year = current_year + ((current_month + i) // 13)
            date_str = datetime(year, month, 1).strftime("%b %y")
            price = round(base.get(name.capitalize(), 1000) * (1 + (i % 6) * 0.02), 2)
            change = round((i % 6) * 0.5, 2)
            crop_price.append([date_str, price, change])
            
        return max_crop, min_crop, crop_price

def TwelveMonthPrevious(name):
    try:
        name = name.lower()
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        # Find the commodity
        commodity = None
        for i in commodity_list:
            if name == str(i).lower():
                commodity = i
                break
                
        if not commodity:
            # Return dummy data if commodity not found
            crop_price = []
            for i in range(12):
                prev_month = (current_month - i - 1) % 12 or 12
                prev_year = current_year - ((i + 12 - current_month) // 12)
                date_str = datetime(prev_year, prev_month, 1).strftime("%b %y")
                price = round(base.get(name.capitalize(), 1000) * (1 - (i % 6) * 0.01), 2)
                crop_price.append([date_str, price])
            return crop_price
            
        month_with_year = []
        for i in range(1, 13):
            prev_month = (current_month - i) % 12 or 12
            prev_year = current_year - ((i + 12 - current_month) // 12)
            prev_rainfall = annual_rainfall[prev_month - 1]
            month_with_year.append((prev_month, prev_year, prev_rainfall))
            
        wpis = []
        for m, y, r in month_with_year:
            current_predict = commodity.getPredictedValue([float(m), y, r])
            wpis.append(current_predict)

        # Get the base price for the commodity
        base_price = base.get(name.capitalize(), 1000)
        
        crop_price = []
        for i in range(len(wpis)):
            m, y, r = month_with_year[i]
            x = datetime(y, m, 1)
            x = x.strftime("%b %y")
            crop_price.append([x, round((wpis[i] * base_price) / 100, 2)])
            
        return crop_price
    except Exception as e:
        print(f"Error in TwelveMonthPrevious for {name}: {e}")
        # Return dummy data on error
        crop_price = []
        for i in range(12):
            prev_month = (current_month - i - 1) % 12 or 12
            prev_year = current_year - ((i + 12 - current_month) // 12)
            date_str = datetime(prev_year, prev_month, 1).strftime("%b %y")
            price = round(base.get(name.capitalize(), 1000) * (1 - (i % 6) * 0.01), 2)
            crop_price.append([date_str, price])
        return crop_price

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            account.create(user_id='unique()', email=email, password=password)
            return redirect('/login')
        except AppwriteException as e:
            return f"Signup Error: {e.message}"
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            session_response = account.create_email_password_session(email, password)
            session['email'] = email
            # return f"Logged in as {email} <br><a href='/logout'>Logout</a>"
            return render_template('index1.html', email=email)

        except AppwriteException as e:
            return f"Login Error: {e.message}"
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

def load_csv_from_appwrite(file_id):
    try:
        file_bytes = storage.get_file_download(app.config['APPWRITE_BUCKET_ID'], file_id)
        csv_content = io.BytesIO(file_bytes)
        df = pd.read_csv(csv_content)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

# Initialize the Appwrite client and load commodities when the app starts
# client = initialize_appwrite_client()

# Load all commodities at startup
try:
    for file_id in commodity_dict.values():
        commodity_list.append(Commodity(file_id=file_id, client=client))
except Exception as e:
    print(f"Error loading commodities: {e}")
    # Initialize with empty list if loading fails
    commodity_list = []

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == "__main__":
    # For local development
    if not os.environ.get('VERCEL_ENV') == 'production':
        app.run(debug=True, host='0.0.0.0', port=5000)