numbers = {"ONE":1,
           "TWO":2,
           "THREE":3,
           "FOUR":4,
           "FIVE":5,
           "SIX":6,
           "SEVEN":7,
           "EIGHT":8,
           "NINE":9,
           "TEN":10,
           "ELEVEN":11,
           "TWELVE":12,
           "THIRTEEN":13,
           "FOURTEEN":14,
           "FIFTEEN":15,
           "SIXTEEN":16,
           "SEVENTEEN":17,
           "EIGHTEEN":18}

with open("sim_data.csv", 'r') as datafile:
    with open("correct.csv", 'w') as outfile:
        headers = datafile.readline()
        outfile.write("t,current,current_mag,previous,previous_mag,addition,addition_mag,recall,recall_mag\n")
        current_input = ""
        previous_input = ""
        addition_output = ""

        for line in datafile:
            data = [x.strip() for x in line.split(',')]
            t = float(data[0])
            input = data[1]
            input_mag = float(data[2])
            current = data[3]
            current_mag = float(data[4])
            previous = data[5]
            previous_mag = float(data[6])
            addition = data[7]
            addition_mag = float(data[8])
            recall = data[9]
            recall_mag = float(data[10])

            if (t % 1.0) > 0.0041 and (t % 1.0) < 0.0059:
                previous_input = current_input
                current_input = input
                if previous_input != "" and current_input != "":
                    add_num = numbers[current_input] + numbers[previous_input]
                    addition_output = [key for key, value in numbers.items() if value == add_num][0]
                else:
                    addition_output = ""
            
            cur_mag = 0
            if current == current_input:
                cur_mag = current_mag

            prev_mag = 0
            if previous == previous_input:
                prev_mag = previous_mag

            add_mag = 0
            if addition == addition_output:
                add_mag = addition_mag

            rec_mag = 0
            if recall == addition_output:
                rec_mag = recall_mag
            
            outfile.write("%f,%s,%f,%s,%f,%s,%f,%s,%f\n" % (t, current_input, cur_mag, previous_input, prev_mag, addition_output, add_mag, addition_output, rec_mag))
