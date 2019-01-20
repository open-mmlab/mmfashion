import numpy


def count_attribute(label_file):
    labels = numpy.loadtxt(label_file)
    num_lines = sum(1 for line in open(label_file))
    print("%d images in testing dataset" % num_lines)

    # sum the column
    column_sum = numpy.zeros(88)
    for row in labels:
        for i, label in enumerate(row):
            column_sum[i] += row[i]
            #        print(len(row))
    return column_sum, num_lines
