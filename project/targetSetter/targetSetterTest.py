# importing sys

import unittest
from decisionTree import Tree





class TestClass(unittest.TestCase):

    def test_class10008(self):
        input=[100008,0.383147,0.616853,0,0,0.616853,0.038452149,0.578400851,0.418397819,0.198455181,0,0.104752126,0.512100874,0,0.054453,0.945547,0.201462524,0.181684476,0,0,0.0272265,0,0.0272265,0,0,0,0,0,0,0.279952491,0.138445328,0,0,0.092885571,0,0,0,0.325512248]
        id=str(input[0])
        answer=(list(map(str,input[1:])))
        decisionTree=Tree()
        currentClass=decisionTree.computeTargetValue(id,answer)

        expected="3"

        self.assertTrue(currentClass.target==expected)

    def test_class10043(self):
        input=[100143,0.269843,0.730157,0,0.730157,0,0,0,0,0,0,0,0,0,0.410635,0.589365,0,0,0.269843,0,0,0.034116788,0,0.136878196,0.239640016,0,0.56142867,0,0.16872833,0,0,0,0,0,0,0,0,0]
        id=str(input[0])
        answer=(list(map(str,input[1:])))
        decisionTree=Tree()
        currentClass=decisionTree.computeTargetValue(id,answer)

        expected="2"

        self.assertTrue(currentClass.target==expected)


    def test_class10134(self):
        input=[100134,0.021834,0.976952,0.001214,0.021750859,0.955201141,0.313076726,0.642124415,0.546490632,0.408710509,0.160096487,0.760687801,0.034416852,0,0.611499,0.388501,0.010917,0.010917,0,0,0.032377038,0.064143188,0.4502247,0,0.032377038,0.032377038,0,0,0.021750859,0.207252747,0.152044077,0.187193808,0,0,0.054649063,0.081973595,0.081973595,0.327894379]
        id=str(input[0])
        answer=(list(map(str,input[1:])))
        decisionTree=Tree()
        currentClass=decisionTree.computeTargetValue(id,answer)

        expected="3"

        self.assertTrue(currentClass.target==expected)
        

    def test_class10123(self):
        input=[100123,0.462492,0.456033,0.081475,0,0.456033,0,0.456033,0,0.456033,0,0,0.456033,0,0.687647,0.312353,0.388157511,0.074334489,0,0,0,0,0,0.213858217,0.473788783,0,0,0,0,0,0,0,0,0,0,0,0,0]
        id=str(input[0])
        answer=(list(map(str,input[1:])))
        decisionTree=Tree() 
        currentClass=decisionTree.computeTargetValue(id,answer)

        expected="1"

        self.assertTrue(currentClass.target==expected)

    def test_class10445(self):
        input=[100445,0.186593,0.796066,0.017341,0,0.796066,0,0.796066,0.355462575,0.440603425,0.135023142,0.405012111,0.220226086,0.03580466,0.262162,0.737838,0,0.186593,0,0.074903354,0,0.074903354,0.074903354,0,0.037451677,0,0,0,0,0.141345783,0.214116792,0,0,0.164982622,0,0,0,0.190479953]
        id=str(input[0])
        answer=(list(map(str,input[1:])))
        decisionTree=Tree() 
        currentClass=decisionTree.computeTargetValue(id,answer)

        expected="4"

        self.assertTrue(currentClass.target==expected)


    def teat_class0(self):
        input=[135453,0.360342,0.098619,0.541039,0,0.098619,0,0.098619,0,0.098619,0,0,0,0.098619,0.12106,0.87894,0.256566747,0.103775253,0,0,0,0.12106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        id=str(input[0])
        answer=(list(map(str,input[1:])))
        decisionTree=Tree() 
        currentClass=decisionTree.computeTargetValue(id,answer)

        expected="0"

        self.assertTrue(currentClass.target==expected)


if __name__=='__main__':
        unittest.main()
