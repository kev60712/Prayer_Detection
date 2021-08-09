import requests
from datetime import datetime
import pytz

# API Format
# http://zenbo.pythonanywhere.com/api/v1/resources/prayertime?day=1&month=1&hour=21&minute=46&country=AU

def check_islamic_calendar():
    tz_Dubai = pytz.timezone('Asia/Dubai') 
    datetime_Dubai = datetime.now(tz_Dubai)
    #params = {'month':datetime_Dubai.month, 'day':datetime_Dubai.day,'hour':datetime_Dubai.hour, 'minute':datetime_Dubai.minute, 'country':'AU'}
    params = {'month':1, 'day':1, 'hour':21, 'minute':46, 'country':'AU'} # Return True
    
    r = requests.get('http://zenbo.pythonanywhere.com/api/v1/resources/prayertime', params = params)
    print("Dubai time:", datetime_Dubai.strftime("%H:%M:%S"))
    
    if r.text == 'True':
        return True
    else:
        return False
    
if __name__ == '__main__':
    check_islamic_calendar()
