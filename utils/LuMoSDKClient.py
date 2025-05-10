import zmq
# import LusterFrameStruct_pb2
from . import LusterFrameStruct_pb2

class LusterFrame:
    def __init__(self):
        self.context = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)

    def Connnect(self, ip):
        self.subscriber.setsockopt(zmq.CONFLATE, 1)
        connectIp = "tcp://" + ip + ":6868"
        self.subscriber.connect(connectIp)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    def Close(self):
        self.subscriber.close()

    def ReceiveData(self, flag):
        frame = LusterFrameStruct_pb2.Frame()
        if flag == 0:
            message = self.subscriber.recv()  
        elif flag == 1:
            try:
                message = self.subscriber.recv(zmq.DONTWAIT)
            except zmq.Again:
                return
        else:
            return
        frame.ParseFromString(message)
        return frame