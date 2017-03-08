import sys
import os

if os.getenv("COMPUTERNAME") == "DESKTOP-RJA3ECD":
    sys.path.append("C:/Users/student/taguchi/python")
else:
    sys.path.append("C:/Users/kotaro/Documents/Lab/python")


from postprograms import postprocessing_stable

def main():
    postprocessing_stable.postprocess_clean()

if __name__ == '__main__':
    main()
