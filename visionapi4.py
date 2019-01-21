from google.cloud import vision
from google.protobuf import json_format
from flask import jsonify
# Supported mime_types are: 'application/pdf' and 'image/tiff'
mime_type = 'application/pdf'
from google.oauth2 import service_account
import sys
import pandas as pd
import re
import logging

log_file_name = "test.log"
logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")

credentials = service_account.Credentials.from_service_account_file('key.json')

client = vision.ImageAnnotatorClient(credentials=credentials)
# How many pages should be grouped into each json output file.
batch_size = 2

def get_line_lists(file_path):
    feature = vision.types.Feature(type = vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION)
    with open(file_path, 'rb') as image_file :
        content = image_file.read()
    image = vision.types.Image(content = content)
    response = client.document_text_detection(image = image)
    line_list = str(response.text_annotations[0].description).split("\n")
    #print("line list : {0}".format(line_list))
    line_list = [i for i in line_list if len(i) > 1]
    #logging.debug("\tline_list : {0}".format(line_list))
    return line_list

keywords_field_name = ["umrn", "date", "nachtype", "bankcode", "utilitycode", "authorizebank", "debitaccounttype", "acnumber", "withbank", "ifsc", "micr", "amount" , \
"amountinwords", "frequency", "debittype", "reference1", "reference2", "mobile", "email", "fromdate", "todate", "untilcancelled", "issigned1", "issigned2", "issigned3"]


def ret_json(sample_line_list):
    fin_json = {}
    logging.debug("\tret_json function entered.")
    #logging.debug("\tinput sample_line_list : {0}".format(sample_line_list))
    for i in sample_line_list:
        logging.debug("\ti in sample_line_list : {0}".format(i))
        umrn_regex = re.search(r'um.*n.*', i, re.IGNORECASE)
        if "umrn" in i.lower() or umrn_regex:
            logging.debug("\tumrn in i.lower()")
            text_wo_umrn = i.lower()[i.lower().find('umrn') + 4 :]
            logging.debug("\ttext_wo_umrn : {0}".format(text_wo_umrn))
            if "Date" in text_wo_umrn:
                logging.debug("\tDate in text_wo_umrn.")
                umrn_date_regex = re.findall(r'\d', text_wo_umrn , re.IGNORECASE)
                logging.debug("\tdate_regex : {0}".format(umrn_date_regex))
                if umrn_date_regex:
                    fin_json['date'] = "".join(umrn_date_regex) 
                    logging.debug("\tfin_json['date'] = {0}".format(fin_json['date']))
        if "date" in i.lower():
            logging.debug("\tdate in i.lower().")
            date_regex = re.findall(r'\d', i.lower() , re.IGNORECASE)
            logging.debug("\tdate_regex : {0}".format(date_regex))
            if date_regex:
                fin_json['date'] = "".join(date_regex) 
                logging.debug("\tfin_json['date'] = {0}".format(fin_json['date']))

        bankcode_regex = re.search(r'[B|G][ANK](.*)[C|G][ode]', i, re.IGNORECASE)
        logging.debug("\tbankcode_regex : {0}".format(bankcode_regex))
        utilitycode_regex = re.search(r'[U][t](.*)[C][o](.*)[N][A](.*)', i , re.IGNORECASE)
        logging.debug("\tutilitycode_regex : {0}".format(utilitycode_regex))
        if not utilitycode_regex:
            utilitycode_regex = re.search(r'[itil](.*)[C]od(.*)', i, re.IGNORECASE)
            logging.debug("\tutilitycode_regex second : {0}".format(utilitycode_regex))
        if bankcode_regex and utilitycode_regex:
            logging.debug("\tif bankcode_regex and utilitycode_regex entered.")
            bankcode_utility_find = re.search(r'[B|G]ank.*[C|G]ode\s(.*)\sUt.*Co.*\s[NA](.*)', i, re.IGNORECASE)
            logging.debug("\tbankcode_utility_find : {0}".format(bankcode_utility_find))
            if not bankcode_utility_find:
                bankcode_utility_find = re.search(r'[B|G]ank.*[C|G]ode\s(.*)\silit.*Co.*\s[NA](.*)', i, re.IGNORECASE)
                logging.debug("\tbankcode_utility_find second regex : {0}".format(bankcode_utility_find))
            if bankcode_utility_find:
                fin_json["bankcode"] = bankcode_utility_find[1]
                logging.debug("\tfin_json['bankcode'] = {0}".format(bankcode_utility_find[1]))
                utilitycode_nums = bankcode_utility_find[2]
                logging.debug("\tutilitycode_nums : {0}".format(utilitycode_nums))
                utilitycode_num_find = re.findall(r'\d', utilitycode_nums.lower() , re.IGNORECASE)
                logging.debug("\tutilitycode_num_find : {0}".format(utilitycode_num_find))
                utilitycode_final = "NACH" + "".join(utilitycode_num_find)
                logging.debug("\tutilitycode_final : {0}".format(utilitycode_final))
                fin_json["utilitycode"] = utilitycode_final
        if bankcode_regex and not utilitycode_regex:
            bankcode_find = re.search(r'[B|G]ank.*[C|G]ode\s(.*)', i, re.IGNORECASE)
            logging.debug("\tbankcode_find : {0}".format(bankcode_find))
            if bankcode_find:
                fin_json["bankcode"] = bankcode_find[1]
                logging.debug("\tfin_json['bankcode'] = {0}".format(bankcode_find[1]))
        if not bankcode_regex and utilitycode_regex:
            utilitycode_find = re.search(r'[itil].*[C|G|O]ode\s(.*)', i, re.IGNORECASE)
            logging.debug("\tutilitycode_find : {0}".format(utilitycode_find))
            if utilitycode_find:
                fin_json["utilitycode"] = utilitycode_find[1]
                logging.debug("\tfin_json['utilitycode'] = {0}".format(utilitycode_find[1]))

        authorizebank_regex = re.search(r'[W].*\s[h|n|r][e|o|c].*\s[Au].*[e]\s(.*)', i, re.IGNORECASE)
        logging.debug("\tauthorizebank_regex : {0}".format(authorizebank_regex))
        if not authorizebank_regex:
            authorizebank_regex = re.search(r'[w|vv].*\s[h|n].*\s[auth|aut].*e(.*)', i, re.IGNORECASE)
            logging.debug("\tauthorizebank_regex second : {0}".format(authorizebank_regex))
        if authorizebank_regex and "deb" not in i.lower():
            logging.debug("\tauthorizebank_regex successful and 'deb' is not in i.lower().")
            if "authorizebank" not in fin_json or len(fin_json["authorizebank"]) < 3:
                fin_json['authorizebank'] = authorizebank_regex[1]
                logging.debug("\tfin_json['authorizebank'] : {0}".format(fin_json['authorizebank']))
        if authorizebank_regex and "deb" in i.lower():
            logging.debug("\tauthorizebank_regex successful but 'deb' is in i.lower().")
            authbank_todebit_regex = re.search(r'[W].*\s[h|n|r][e|o|c].*\s[Au].*[e]\s(.*)[ed]\s(.*)', i, re.IGNORECASE)
            logging.debug("\tauthbank_todebit_regex : {0}".format(authbank_todebit_regex))
            if authbank_todebit_regex:
                fin_json["authorizebank"] = authbank_todebit_regex[1]
                logging.debug("\tfin_json['authorizebank'] : {0}".format(authbank_todebit_regex[1]))
                todebit_phrase = authbank_todebit_regex[2]
                logging.debug("\ttodebit_phrase : {0}".format(todebit_phrase))
                if "SB" in todebit_phrase:
                    fin_json["debitaccounttype"] = "SB"
                    logging.debug("\t'SB' in todebit_phrase.")
                if "CC" in todebit_phrase:
                    fin_json["debitaccounttype"] = "CC"
                    logging.debug("\t'CC' in todebit_phrase.")
                if "CA" in todebit_phrase:
                    fin_json["debitaccounttype"] = "CA"
                    logging.debug("\t'CA' in todebit_phrase.")


        if "CREATE" in i.upper():
            logging.debug("\t'CREATE' in i.upper()")
            print(i)
            except_create_text = i[i.upper().find('CREATE') + 6:]
            logging.debug("\texcept_create_text : {0}".format(except_create_text))
            create_check_box_text = ""
            if len(except_create_text) >2:
                create_check_box_text = except_create_text.split()[0]            
            logging.debug("\tcreate_check_box_text : {0}".format(create_check_box_text))
            if create_check_box_text == "2" or create_check_box_text == "?":
                fin_json["nachtype"] = "CREATE"
                logging.debug("\tcreate_check_box_text is either '2' or '?'.")

        if "MODIFY" in i.upper():
            logging.debug("\tMODIFY is in i.upper().")
            modify_check_box_text = i[i.upper().find('MODIFY') + 6:].split()[0]
            logging.debug("\tmodify_check_box_text : {0}".format(modify_check_box_text))
            if modify_check_box_text == "O" or modify_check_box_text == "D":
                modify_check = False
                logging.debug("\tmodify_check_box_text is either 'O' or 'D'; meaning modify is unchecked.")
            elif modify_check_box_text == "2":
                modify_check = True
                fin_json["nachtype"] = "MODIFY"
                logging.debug("\tmodify_check_box_text is '2'; meaning modify is checked.")

        if "CANCEL" in i.upper():
            logging.debug("\tCANCEL is in i.upper()")
            cancel_check_box_text = i[i.upper().find('CANCEl') + 6:].split()[0]
            logging.debug("\tcancel_check_box_text : {0}".format(cancel_check_box_text))
            if cancel_check_box_text == "O" or cancel_check_box_text == "D":
                cancel_check = False
                logging.debug("\tcancel_check_box_text is either 'O' or 'D'; meaning CANCEL is unchecked.")
            elif cancel_check_box_text == "2":
                cancel_check = True
                fin_json["nachtype"] = "CANCEL"
                logging.debug("\tcancel_check_box_text is '2'; meaning CANCEl box is checked.")

        if "cancel_check" in locals() and "modify_check" in locals():
            logging.debug("\tcancel_check and modify_check variables are in locals.")
            if cancel_check == False and modify_check == False:
                fin_json["nachtype"] = "CREATE"
                logging.debug("\tcancel_check and modify_check both are False.")


        acnumber_regex = re.search(r'[B|G][ank].*a.*c.*umbe(.*)', i, re.IGNORECASE)
        logging.debug("\tacnumber_regex : {0}".format(acnumber_regex))
        if acnumber_regex:
            logging.debug("\tacnumber_regex is successful.")
            acnum_text = acnumber_regex[1]
            logging.debug("\tacnum_text : {0}".format(acnum_text))
            acnum_digits = re.findall(r'\d', acnum_text , re.IGNORECASE)
            logging.debug("\tacnum_digits : {0}".format(acnum_digits))
            acnum = "".join(acnum_digits)
            logging.debug("\tacnum : {0}".format(acnum))
            if "acnumber" not in fin_json:
                logging.debug("\tacnumber not in fin_json.")
                fin_json["acnumber"] = acnum
            elif "acnumber" in fin_json:
                if len(fin_json['acnumber']) < 3:
                    logging.debug("\tacnumber is in fin_json but length of it is <3.")
                    fin_json["acnumber"] = acnum


        withbank_regex = re.search(r'[with]\s[B|G]ank(.*)', i, re.IGNORECASE)
        logging.debug("\twithbank_regex : {0}".format(withbank_regex))
        logging.debug("\twithbank_regex is successful, but it may contain ifsc also.")
        withbank_ifsc_regex = re.search(r'[with]\s[B|G]ank(.*)ifsc(.*)', i, re.IGNORECASE)
        logging.debug("\twithbank_ifsc_regex : {0}".format(withbank_ifsc_regex))
        logging.debug("\twithbank_ifsc_regex is successful, meaning ifsc is in with withbank")
        if withbank_regex and not withbank_ifsc_regex:
            logging.debug("\twithbank_regex is successful but withbank_ifsc_regex is not, means ifsc is not in with withbank.")
            fin_json['withbank'] = withbank_regex[1]
        if withbank_regex and withbank_ifsc_regex:
            logging.debug("\twithbank_regex is successful, withbank_ifsc_regex is also successful; meaning ifsc is in with withbank.")
            logging.debug("\tin this case, possibility is that micr is also with withbank and ifsc in same line. so we introduce another regex for micr find.")
            withbank_ifsc_micr_regex = re.search(r'[with]\s[B|G]ank(.*)ifsc(.*)m.*cr(.*)', i, re.IGNORECASE)
            logging.debug("\twithbank_ifsc_micr_regex : {0}".format(withbank_ifsc_micr_regex))
            if withbank_ifsc_micr_regex:
                logging.debug("\twithbank_ifsc_micr_regex successful.")
                fin_json['ifsc'] = withbank_ifsc_micr_regex[2]
                logging.debug("\tfin_json['ifsc'] = withbank_ifsc_micr_regex[2] : {0}".format(withbank_ifsc_micr_regex[2]))
                fin_json['micr'] = withbank_ifsc_micr_regex[3]
                logging.debug("\tfin_json['micr'] = withbank_ifsc_micr_regex[3] : {0}".format(withbank_ifsc_micr_regex[3]))
            if not withbank_ifsc_micr_regex:
                logging.debug("\twithbank_ifsc_micr_regex is not successful, so we get ifsc from withbank_ifsc_regex.")
                fin_json['ifsc'] = withbank_ifsc_regex[2]


        ifsc_micr_regex = re.search(r'ifsc(.*)m.*cr(.*)', i, re.IGNORECASE)
        logging.debug("\tifsc_micr_regex is successful.")
        if ifsc_micr_regex and not withbank_regex:
            logging.debug("\tifsc_micr_regex is successful and withbank_regex is not successful, means only ifsc and micr are in same line but not with withbank.")
            fin_json["ifsc"] = ifsc_micr_regex[1]
            logging.debug("\tfin_json['ifsc'] = ifsc_micr_regex[1] : {0}".format(ifsc_micr_regex[1]))
            fin_json['micr'] = ifsc_micr_regex[2]
            logging.debug("\tfin_json['micr'] = ifsc_micr_regex[2] : {0}".format(ifsc_micr_regex[2]))

        amountinwords_regex = re.search(r'.*amount\s.*rupees\s(.*)', i, re.IGNORECASE)
        logging.debug("\tamountinwords_regex is successful, meaning amountinwords is in the line, not sure about amount number though.")
        amountinwords_amount_regex = re.search(r'.*amount\s.*rupees\s(.*)[\d{,9}]', i, re.IGNORECASE)
        logging.debug("\tamountinwords_amount_regex is successful, meaning amountinwords along with amount in number is present in same line.")
        if amountinwords_regex and not amountinwords_amount_regex:
            logging.debug("\tonly amountinwords is in the line.")
            if "amountinwords" not in fin_json:
                logging.debug("\tamountinwords not in fin_json")
                fin_json["amountinwords"] = amountinwords_regex[1]
            elif "amountinwords" in fin_json:
                if len(fin_json["amountinwords"]) < 3:
                    logging.debug("\tamountinwords is in fin_json but the length is < 3.")
                    fin_json["amountinwords"] = amountinwords_regex[1]
        if amountinwords_regex and amountinwords_amount_regex:
            logging.debug("\tamountinwords along with amount number is present in same line.")
            amountinwords_text = amountinwords_regex[1]
            logging.debug("\tamountinwords_text : {0}".format(amountinwords_text))
            amount_in_words = "".join([i for i in amountinwords_text if not i.isnumeric()])
            logging.debug("\tamount_in_words : {0}".format(amount_in_words))
            if "amountinwords" not in fin_json:
                fin_json["amountinwords"] = amount_in_words
            elif "amountinwords" in fin_json:
                if len(fin_json["amountinwords"]) < 3:
                    fin_json["amountinwords"] = amount_in_words
            amount_num = "".join([i for i in amountinwords_text if not i.isalpha()])
            logging.debug("\tamount_num : {0}".format(amount_num))
            if "amount" not in fin_json:
                logging.debug("\tamount not in fin_json.")
                fin_json["amount"] = amount_num
            elif "amount" in fin_json:
                if len(fin_json["amount"].replace(" ","")) < 3:
                    logging.debug("\tamount is in fin_json but the length is < 3.")
                    fin_json["amount"] = amount_num

        if amountinwords_regex and not amountinwords_amount_regex:
            logging.debug("\tonly amountinwords is in the line, checking second time.")
            amountinwords_next_line = sample_line_list[sample_line_list.index(i) +1]
            logging.debug("\tamountinwords_next_line : {0}".format(amountinwords_next_line))
            amount_regex = re.search(r'[\d]{,3},[\d]{,3}.[\d]{,3}', amountinwords_next_line)
            logging.debug("\tamount_regex : {0}".format(amount_regex))
            if not amount_regex:
                amount_regex = re.search(r'[\d]{,8}', amountinwords_next_line)
                logging.debug("\tamount_regex second one : {0}".format(amount_regex))
            if amount_regex:
                logging.debug("\tamount_regex is successful.")
                fin_json["amount"] = amount_regex[0]
                logging.debug("\tfin_json['amoun'] = amount_regex[0] : {0}".format(amount_regex[0]))


        reference1_regex = re.search(r'.*ref.*[1|?|\||L|l]\s(.*)', i, re.IGNORECASE)
        logging.debug("\treference1_regex : {0}".format(reference1_regex))
        reference1_mobile_regex = re.search(r'.*ref.*[1|?|\||L|l]\s(.*)mob.*[\d]{,11}', i, re.IGNORECASE)
        logging.debug("\treference1_mobile_regex : {0}".format(reference1_mobile_regex))
        if reference1_regex and not reference1_mobile_regex:
            logging.debug("\treference1_regex is successful but not reference1_mobile_regex; meaning only reference1 is in this line.")
            if "reference1" not in fin_json:
                fin_json["reference1"] = reference1_regex[1]
            elif "reference1" in fin_json:
                if len(fin_json["reference1"]) < 3:
                    fin_json["reference1"] = reference1_regex[1]


        mobile_regex = re.search(r'.*mob.*\sn(.*)\d{,11}' , i, re.IGNORECASE)
        logging.debug("\tmobile_regex : {0}".format(mobile_regex))
        if mobile_regex:
            mobile_text = mobile_regex[1]
            logging.debug("\tmobile_text : {0}".format(mobile_text))
            mobile_num = "".join([i for i in mobile_text if i.isnumeric()])
            logging.debug("\tmobile_num : {0}".format(mobile_num))
            if "mobile" not in fin_json:
                fin_json["mobile"] = mobile_num
            elif "mobile" in fin_json:
                if len(fin_json["mobile"]) < 3:
                    fin_json["mobile"] = mobile_num

        """
        if "FREQ" in i.upper() or "frequency" in i.lower():
            keywords_pos["frequency"] = sample_line_list.index(i)
        reference2_regex = re.search(r'(.*)[ref](.*)[2](.*)', i, re.IGNORECASE)
        if reference2_regex:
            keywords_pos["reference2"] = sample_line_list.index(i)
        fromdate_regex = re.search(r'[From]\s\d{2,}(.*)', i, re.IGNORECASE)
        if fromdate_regex:
            keywords_pos["fromdate"] = sample_line_list.index(i)
    keywords_pos = sorted(keywords_pos.items(), key=lambda kv: kv[1])
    print(keywords_pos)
    return keywords_pos
    """
    print(fin_json)
    return fin_json

ret_json(get_line_lists(sys.argv[1]))
