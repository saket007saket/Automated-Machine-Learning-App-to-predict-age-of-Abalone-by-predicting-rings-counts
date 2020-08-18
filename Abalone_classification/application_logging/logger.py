
######----------this file is used to creat log file which will be handy to find in which
#     file and at what time app failed so that we can resolve the problem


from datetime import datetime


class App_Logger:
    def __init__(self):
        pass

    def log(self, file_object, log_message):
        self.now = datetime.now() # we r finding here datetime of the situation which is running
        self.date = self.now.date() # from that datetime we r extracting date
        self.current_time = self.now.strftime("%H:%M:%S") # from that date time which has stored in now from that
         # datetime we r extracting hour min and sec as time
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")

        # now we r storing and writing the log message to as at what date its failed , at what time it failed , and what is the log message
