import sys
import os

if os.getenv("COMPUTERNAME") == "DESKTOP-RJA3ECD":
    sys.path.append("C:/Users/student/taguchi/python")
else:
    sys.path.append("C:/Users/kotaro/Documents/Lab/python")


from postprograms import postprocessing

def main():
    postprocessing.postprocess_clean()

if __name__ == '__main__':
    main()
