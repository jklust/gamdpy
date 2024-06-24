import unittest

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def test_(self):
        print('test_bare')
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
