from prepare_data import parse_annotations
import glob

BEG_SYL, END_SYL, BEG_GLOT, END_GLOT, F0, OQ_1, OQ_2, OQ_3, OQ_4, OQ_MC, CREAK = range(11)
def check_segmentation_errors():
    files = glob.glob("/home/mcoavoux/data/phd_corpus/*.xlsx")
    
    for f in files:
        print(f)
        base_f = f.split("/")[-1]
        annotations = parse_annotations(f)
        for ID in annotations:
            for line in annotations[ID]:
                assert(line[BEG_GLOT] < line[END_GLOT])
                if line[END_SYL] < line[BEG_SYL] + line[END_GLOT]:
                    print(base_f, ID)
                    break

def check_peak_det0():
    files = glob.glob("/home/mcoavoux/data/phd_corpus/*.xlsx")
    
    for f in files:
        base_f = f.split("/")[-1]
        annotations = parse_annotations(f)
        for ID in annotations:
            for line in annotations[ID]:
                if line[OQ_1] == 0:
                    values = line[OQ_1:OQ_MC+1]
                    assert(all([v == 0 for v in values]))
                    print(f.split("/")[-1], ID)

if __name__ == "__main__":
    #check_segmentation_errors()
    check_peak_det0()