from google.cloud import vision
from google.protobuf import json_format
from flask import jsonify
# Supported mime_types are: 'application/pdf' and 'image/tiff'
mime_type = 'application/pdf'
from google.oauth2 import service_account
import sys, json, re, os, logging
import pandas as pd
from pprint import pprint
import os.path
from tess_ang_check import run_tesseract, parse_hocr, rotate_image
from t2n2t import num2word, word2num
from difflib import SequenceMatcher

file_name = sys.argv[1]
unique_log_file = os.path.splitext(file_name)[0].split("/")[-1] + ".log"
#log_file_name = "test.log"
logging.basicConfig(filename = unique_log_file, level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")

output_img_filename = ".".join(file_name.split(".")[:-1]) + ".rot." + file_name.split(".")[-1]
print(output_img_filename)
logging.debug("\toutput_img_filename : {0}".format(output_img_filename))
#output_hocr_filename =  ".".join(output_img_filename.split(".")[:-1])
output_hocr_filename = ".".join(file_name.split(".")[:-1])
print(output_hocr_filename)
logging.debug("\toutput_hocr_filename : {0}".format(output_hocr_filename))

run_tesseract(file_name, output_hocr_filename, "eng")
print("tesseract done.")
logging.debug("\trunning tesseract is done.")
check_rot = rotate_image(cv_image = file_name, angle = parse_hocr(hocr_file = output_hocr_filename  + ".hocr"), file_name_rot = output_img_filename)

credentials = service_account.Credentials.from_service_account_file('key.json')

client = vision.ImageAnnotatorClient(credentials=credentials)
# How many pages should be grouped into each json output file.
batch_size = 2
def removespecialcharacter(string1):
    string2=re.sub('[^A-Za-z0-9\.\,\s]+', '', string1)
    return string2
def onlyalphanumeric(string1):
    string2 = re.sub('[^A-Za-z0-9]+', '', string1)
    return string2
def removespace(string1):
    string2=re.sub('\s+', '', string1)
    return string2
def onlyalphabetic(string1):
    string2 = re.sub('[^A-Za-z\s]+', '', string1)
    return string2
def check_string_similarity_list(str1, list1):
    logging.debug("\tchecking similar string : {0}, inside a list : {1}".format(str1, list1))
    if list1:
        similarity_dict = {}
        for i in list1:
            similarity_dict[i] = SequenceMatcher(a=str1.lower(), b=i.lower()).ratio()
        sorted_similarity_dict = sorted(similarity_dict.items(), key = lambda kv: kv[1], reverse = True)
        highly_similar_key = next(iter(dict(sorted_similarity_dict)))
        sorted_similarity_dict = dict(sorted_similarity_dict)
        logging.debug("\tfound similar string in the list : {0}".format(highly_similar_key))
        logging.debug("\tsorted_similarity_dict : {0}".format(sorted_similarity_dict))
        return (sorted_similarity_dict ,  highly_similar_key)

def get_line_lists(file_path):
    feature = vision.types.Feature(type = vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION)
    with open(file_path, 'rb') as image_file :
        content = image_file.read()
    image = vision.types.Image(content = content)
    response = client.document_text_detection(image = image)
    #logging.debug("\tresponse : {0}".format(response))
    #logging.debug("\tresponse.text_annotations : {0}".format(response.text_annotations))
    for i in response.text_annotations:
        logging.debug("\ti.description : {0}".format(i.description))
    logging.debug("\tresponse.text_annotations[0].description : {0}".format(response.text_annotations[0].description))
    line_list = str(response.text_annotations[0].description).split("\n")
    #print("line list : {0}".format(line_list))
    line_list = [i for i in line_list if len(i) > 1]
    #logging.debug("\tline_list : {0}".format(line_list))
    return line_list

keywords_field_name = ["status", "umrn", "date", "nachtype", "bankcode", "utilitycode", "authorizebank", "debitaccounttype", "acnumber", "withbank", "ifsc", "micr", "amount" , \
"amountinwords", "frequency", "debittype", "reference1", "reference2", "mobile", "email", "fromdate", "todate", "untilcancelled", "issigned1", "issigned2", "issigned3"]

ifsc_withbank_dict = {"ICIC" : "ICICI Bank", "SBIN" : "State Bank of India" , "HDFC" : "HDFC Bank", "CITI" : "CITI Bank", "HSBC" : "HSBC Bank", "IBKL" : "IDBI Bank", "IOBA" : "Indian Overseas Bank",\
"BKID" : "Bank of India", "KVBL" : "Karur Vysya Bank", "RBIS" : "Reserve Bank of India", "SYNB" : "Syndicate Bank", "UCBA" : "UCO Bank" , "UBIN" : "Union Bank of India", "UTBI" : "United Bank of India",\
"YESB" : "YES Bank"}

withbank_ifsc_dict = {"icici bank" : "ICIC", "state bank of india": "SBIN", "hdfc bank" : "HDFC", "citi bank" : "CITI", "hsbc bank" : "HSBC", "idbi bank" : "IBKL", "indian overseas bank" : "IOBA",\
"bank of india" : "BKID", "karur vysya bank" : "KVBL", "reserve bank of india" : "RBIS", "syndicate bank" : "SYNB", "uco bank" : "UCBA", "union bank of india" : "UBIN", " united bank of india" : "UTBI",\
"yes bank" : "YESB"}

ifsc_mod_dict = {"ICIC" : ["IC1C" , "1CIC" , "1C1C"] , "HDFC" : ["MDFC", "HOFC", "HDEC", "HDFG"], "CITI" : ["CIT1", "C1TI", "C1T1"], "IOBA" : ["TOBA", "IQBA", "IOPA"], "KVBL" :["KVEL", "KUBL"]}

def ret_json(sample_line_list):
    fin_json = {}
    logging.debug("\tret_json function entered.")
    encoded_line_list = [i.encode("utf-8") for i in sample_line_list]
    pprint(encoded_line_list)
    #logging.debug("\tinput sample_line_list : {0}".format(sample_line_list))
    def find_amount(line_list):
        logging.debug("\tfind_amount function entered with line_list : {0} ".format(line_list))
        amount_keywords = ["one", "two","three", "four", "five", "six", "seven", "eight", "nine", "hundred", "thousand"]
        poss_amountinwords = ""
        poss_amount_fin = ""
        poss_amount = ""
        poss_amount_sec = ""
        for am_line in line_list:
            am_line_low = am_line.lower()
            am_line_low_split = am_line_low.split()
            #logging.debug("\tam_line_low_split : {0}".format(am_line_low_split))
            common_amount_keyword_list = list(set(am_line_low_split).intersection(amount_keywords))
            logging.debug("\tcommon_amount_keyword_list : {0}".format(common_amount_keyword_list))
            if len(common_amount_keyword_list) >=2 :
                if "rs" in am_line_low:
                    poss_amountinwords = am_line[am_line_low.find('rs') +2:]
                else:
                    poss_amountinwords = am_line
            amount_rs_regex = re.search(r'rs.*\s\d{,8}.*', am_line, re.IGNORECASE)
            logging.debug("\tamount_rs_regex : {0}".format(amount_rs_regex))
            if amount_rs_regex:
                amount_dig = re.findall(r'\d', am_line)
                logging.debug("\tamount_dig : {0}".format(amount_dig))
                if amount_dig:
                    poss_amount = "".join(amount_dig)
            amount_word_regex = re.search(r'.*Ru[p|d]ees(.*)', am_line, re.IGNORECASE)
            logging.debug("\tamount_word_regex : {0}".format(amount_word_regex))
            if amount_word_regex:
                amount_only_regex = re.findall(r'(\d+,\d+.\d+)', line_list[line_list.index(am_line) +1])
                logging.debug("\tamount_only_regex : {0}".format(amount_only_regex))
                if not amount_only_regex:
                    amount_only_regex = re.findall(r'\d+.*,\d+.*.\d+', line_list[line_list.index(am_line) +1])
                    logging.debug("\tamount_only_regex second : {0}".format(amount_only_regex))
                if not amount_only_regex:
                    amount_only_regex = re.findall(r'\d', line_list[line_list.index(am_line) +1])
                    logging.debug("\tamount_only_regex third : {0}".format(amount_only_regex))
                if amount_only_regex:
                    if len(amount_only_regex) > 0 :
                        poss_amount_sec = "".join(amount_only_regex)
            poss_amount_fin = poss_amount if len(poss_amount) > len(poss_amount_sec) else poss_amount_sec
        if poss_amount_fin != "" or poss_amountinwords != "":
            return poss_amount_fin, poss_amountinwords

    def find_amount_with_format(line_list):
        logging.debug("\tfind_amount_with_format function entered.")
        poss_amount_only = ""
        for i in line_list:
            amount_regex = re.findall(r'(\d+,\d+.\d+)', i)
            if not amount_regex:
                amount_regex = re.findall(r'\d+.*,\d+.*.\d+', i)
            if amount_regex:
                poss_amount_only = amount_regex[0]
        if poss_amount_only != "":
            return poss_amount_only

    def find_todebit_line_only(line_list):
        logging.debug("\ttodebit_line_only function entered !")
        poss_todebit = ""
        for todb_line in line_list:
            todebit_search = "to deb" in todb_line
            todebit_regex = re.search(r'to.*(?:deb | debit)(.*)', todb_line, re.IGNORECASE)
            logging.debug("\ttodebit_regex : {0}".format(todebit_regex))
            if todebit_regex:
                if poss_todebit == "":
                    poss_todebit = todebit_regex[1]
                else:
                    pass
        if poss_todebit != "":
            return poss_todebit

    def find_reference1_by_maxamount(line_list):
        logging.debug("\tfind_reference1_by_maxamount function entered.")
        poss_ref1 = ""
        for maxam_line in line_list:
            maximum_amount_regex = re.search(r'.*maxim.*\samount', maxam_line, re.IGNORECASE)
            logging.debug("\tmaximum_amount_regex : {0} ".format(maximum_amount_regex))
            if maximum_amount_regex:
                next_line = line_list[line_list.index(maxam_line)+1]
                logging.debug("\tnext_line : {0}".format(next_line))
                if "ence" in next_line.lower():
                    logging.debug("\tProbably reference1 is in the next line.")
                    reference1_digits = re.findall(r'\d', next_line)
                    if reference1_digits:
                        if len(reference1_digits) >4:
                            poss_ref1 = "".join(reference1_digits)
        return poss_ref1

    def find_bankcode_from_nextline(line_list):
        logging.debug("\tfind_bankcode_from_nextline function entered.")
        poss_bankcode = ""
        for i in line_list:
            bankcode_regex = re.search(r'[B|G][ANK](.*)[C|G][ode]', i, re.IGNORECASE)
            logging.debug("\tbankcode_regex : {0}".format(bankcode_regex))
            if bankcode_regex:
                if "HSBC" not in i or "INDIA" not in i or "Utility" not in i or "ility " not in i:
                    bankcode_nextline = line_list[line_list.index(i)+1]
                    logging.debug("\tbankcode_nextline : {0}".format(bankcode_nextline))
                    if "HSBC" in bankcode_nextline or "INDIA" in bankcode_nextline or "1NDIA" in bankcode_nextline or "1ND1A" in bankcode_nextline or "NDIA" in bankcode_nextline:
                        poss_bankcode = "HSBC02INDIA"
        if poss_bankcode != "":
            return poss_bankcode
    
    def find_bankcode_from_sponsorline(line_list):
        logging.debug("\tfind_bankcode_from_sponsorline function entered.")
        poss_bankcode = ""
        for i in line_list:
            if "sponsor" in i.lower():
                hsbcnda_regex = re.search(r'.*hsbc.*nd.*a.*', i, re.IGNORECASE)
                if hsbcnda_regex:
                    poss_bankcode = "HSBC02INDIA"
        if poss_bankcode != "":
            return poss_bankcode

    def find_authbank_by_strmatch(line_list):
        logging.debug("\tfind_authbank_by_fullerton function entered.")
        poss_authbank = ""
        for i in line_list:
            if "authorise" in i.lower():
                if poss_authbank != "":
                    poss_authbank = i[i.lower().find("authorise")+9:]
                    print(poss_authbank)
        if poss_authbank != "":
            return poss_authbank

    def find_authorizebank_from_authoriseline(line_list):
        logging.debug("\tfind_authorizebank_not_from_authorizeline function entered.")
        poss_authbank = ""
        for i in line_list:
            if "authorise" in i.lower():
                auth_previous_line = line_list[line_list.index(i) -1]
                if "FULL" in auth_previous_line.upper() and "LIMITED" in auth_previous_line.upper():
                    poss_authbank = auth_previous_line[auth_previous_line.upper().find("FULL") : auth_previous_line.upper().find("LIMITED")+7]
                elif "FULL" in auth_previous_line.upper() and  "ltd" in auth_previous_line.lower():
                    poss_authbank = auth_previous_line[auth_previous_line.upper().find("FULL") : auth_previous_line.lower().find("ltd") + 3]

        if poss_authbank != "":
            return poss_authbank

    def find_withbank(line_list):
        logging.debug("\tfind_withbank function entered.")
        poss_withbank = ""
        for i in line_list:
            withbank_regex_2 = re.search(r'with\sb[a|e|o|c]n[k|f|t|l|h](.*)', i, re.IGNORECASE)
            if withbank_regex_2:
                poss_withbank = withbank_regex_2[1]
        if poss_withbank != "":
            return poss_withbank

    def find_bank_acnumber(line_list):
        logging.debug("\tfind_bank_acnumber function entered.")
        poss_bankacnum = ""
        for i in line_list:
            banknumber_regex = re.search(r'bank.*numb(.*)', i, re.IGNORECASE)
            if not banknumber_regex:
                banknumber_regex = re.search(r'[b|g]ank.*numb(.*)', i, re.IGNORECASE)
            if not banknumber_regex:
                banknumber_regex = re.search(r'[b|g]ank.*num(.*)', i, re.IGNORECASE)
            if banknumber_regex:
                if banknumber_regex[1] != None:
                    banknum_text = banknumber_regex[1]
                    poss_bankacnum_digs = re.findall(r'\d', banknum_text)
                    poss_bankacnum = "".join(poss_bankacnum_digs)
        if poss_bankacnum != "":
            return poss_bankacnum

    def find_micr_keyword(line_list):
        logging.debug("find_micr_keyword function entered.")
        poss_micr = ""
        for i in line_list:
            if "micr" in i.lower():
                micr_next_text = i[i.lower().find("micr")+4:]
                micr_digs_regex = re.findall(r'\d', micr_next_text)
                if micr_digs_regex:
                    poss_micr = "".join(micr_digs_regex)
        if poss_micr != "":
            return poss_micr

    def find_ifsc_if_withbank_found(line_list, ifsc_key_param) :
        logging.debug('\tfind_ifsc_if_withbank_found function entered.')
        poss_ifsc = ""
        for i in line_list:
            if ifsc_key_param.lower() in i.lower():
                if "micr" in i.lower():
                    ifsc_key_text = i[i.lower().find(ifsc_key_param) + len(ifsc_key_param) : i.lower().rfind("micr")]
                    ifsc_digits = re.findall(r'\d', ifsc_key_text)
                    if ifsc_digits:
                        ifsc_num_part = "".join(ifsc_digits)
                        poss_ifsc = ifsc_key_param + ifsc_num_part
                elif "micr" not in i.lower():
                    ifsc_key_text2 = i[i.lower().find(ifsc_key_param) + len(ifsc_key_param) :]
                    ifsc_digits2 = re.findall(r'\d', ifsc_key_text2)
                    if ifsc_digits2:
                        ifsc_num_part2 = "".join(ifsc_digits2)
                        poss_ifsc = ifsc_key_param + ifsc_num_part2
        if poss_ifsc != "":
            return poss_ifsc


    for i in sample_line_list:
        #logging.debug("\t\n\ni in sample_line_list : {0}".format(i))
        logging.debug("\n\n")
        umrn_regex = re.search(r'um.*n.*', i, re.IGNORECASE)
        if "umrn" in i.lower() or umrn_regex:
            logging.debug("\tumrn in i.lower()")
            text_wo_umrn = i.lower()[i.lower().find('umrn') + 4 :]
            logging.debug("\ttext_wo_umrn : {0}".format(text_wo_umrn))
            if "Date" in text_wo_umrn or "Dato" in text_wo_umrn:
                logging.debug("\tDate or Dato in text_wo_umrn.")
                umrn_date_regex = re.findall(r'\d', text_wo_umrn , re.IGNORECASE)
                logging.debug("\tdate_regex : {0}".format(umrn_date_regex))
                if umrn_date_regex:
                    if "date" not in fin_json:
                        fin_json['date'] = "".join(umrn_date_regex) 
                        logging.debug("\tfin_json['date'] = {0}".format(fin_json['date']))
                    elif "date" in fin_json:
                        if len(fin_json["date"]) <3:
                            fin_json['date'] = "".join(umrn_date_regex) 
                            logging.debug("\tdate is in fin_json but the length is <3.")
        if "date" in i.lower() or "dato" in i.lower():
            logging.debug("\tdate or dato in i.lower().")
            date_regex = re.findall(r'\d', i.lower() , re.IGNORECASE)
            logging.debug("\tdate_regex : {0}".format(date_regex))
            if date_regex:
                if "date" not in fin_json:
                    fin_json['date'] = "".join(date_regex) 
                    logging.debug("\tfin_json['date'] = {0}".format(fin_json['date']))
                elif "date" in fin_json:
                    if len(fin_json["date"]) <3:
                        fin_json["date"] = "".join(date_regex)
                        logging.debug("\tdate is in fin_json with length < 3.")

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
                bankcode_utility_find = re.search(r'[B|G]ank.*[C|G]ode\s(.*)\sU.*y\sCo.*\s[NA](.*)', i, re.IGNORECASE)
                logging.debug("\tbankcode_utility_find : {0}".format(bankcode_utility_find))
            if not bankcode_utility_find:
                bankcode_utility_find = re.search(r'[B|G]ank.*[C|G]ode\s(.*)\silit.*Co.*\s[NA](.*)', i, re.IGNORECASE)
                logging.debug("\tbankcode_utility_find second regex : {0}".format(bankcode_utility_find))
            if not bankcode_utility_find:
                bankcode_utility_find = re.search(r'ba.*co{,3}\s(.*)ut.*co.*[NA](.*)' , i , re.IGNORECASE)
                logging.debug("\tbankcode_utility_find third regex : {0}".format(bankcode_utility_find))
            if bankcode_utility_find:
                fin_json["bankcode"] = removespace(bankcode_utility_find[1])
                logging.debug("\tfin_json['bankcode'] = {0}".format(bankcode_utility_find[1]))
                utilitycode_nums = bankcode_utility_find[2]
                logging.debug("\tutilitycode_nums : {0}".format(utilitycode_nums))
                utilitycode_num_find = re.findall(r'\d', utilitycode_nums.lower() , re.IGNORECASE)
                logging.debug("\tutilitycode_num_find : {0}".format(utilitycode_num_find))
                utilitycode_final = "NACH" + "".join(utilitycode_num_find)
                logging.debug("\tutilitycode_final : {0}".format(utilitycode_final))
                fin_json["utilitycode"] = removespace(utilitycode_final)
        if bankcode_regex and not utilitycode_regex:
            bankcode_find = re.search(r'[B|G]ank.*[C|G]ode\s(.*)', i, re.IGNORECASE)
            logging.debug("\tbankcode_find : {0}".format(bankcode_find))
            if bankcode_find:
                fin_json["bankcode"] = removespace(bankcode_find[1])
                logging.debug("\tfin_json['bankcode'] = {0}".format(bankcode_find[1]))
        if not bankcode_regex and utilitycode_regex:
            utilitycode_find = re.search(r'[itil].*[C|G|O]ode\s(.*)', i, re.IGNORECASE)
            logging.debug("\tutilitycode_find : {0}".format(utilitycode_find))
            if utilitycode_find:
                fin_json["utilitycode"] = removespace(utilitycode_find[1])
                logging.debug("\tfin_json['utilitycode'] = {0}".format(utilitycode_find[1]))
        if bankcode_regex and "bankcode" not in fin_json:
            pass

        authorizebank_regex = re.search(r'[W].*\s[h|n|r][e|o|c].*\s[Au].*[se|ze]\s(.*)', i, re.IGNORECASE)
        logging.debug("\tauthorizebank_regex : {0}".format(authorizebank_regex))
        if not authorizebank_regex:
            authorizebank_regex = re.search(r'[w|vv].*\s[h|n].*\s[auth|aut].*e\s(.*)', i, re.IGNORECASE)
            logging.debug("\tauthorizebank_regex second : {0}".format(authorizebank_regex))
        if not authorizebank_regex:
            authorizebank_regex = re.search(r'[w|vv].*\she.*\saut{,5}(.*)', i, re.IGNORECASE)
            logging.debug("\tauthorizebank_regex third : {0}".format(authorizebank_regex))
        if authorizebank_regex and "deb" not in i.lower():
            logging.debug("\tauthorizebank_regex successful and 'deb' is not in i.lower().")
            if "authorizebank" not in fin_json or len(fin_json["authorizebank"]) < 3:
                if authorizebank_regex[1] != None:
                    fin_json['authorizebank'] = authorizebank_regex[1]
                    logging.debug("\tfin_json['authorizebank'] : {0}".format(fin_json['authorizebank']))
        if authorizebank_regex and "deb" in i.lower():
            logging.debug("\tauthorizebank_regex successful but 'deb' is in i.lower().")
            authbank_todebit_regex = re.search(r'[W].*\s[h|n|r][e|o|c].*\s[Au].*[e]\s(.*)[ed|td]\s(.*)', i, re.IGNORECASE)
            logging.debug("\tauthbank_todebit_regex : {0}, type(authbank_todebit_regex) : {1}".format(authbank_todebit_regex, type(authbank_todebit_regex)))
            if (authbank_todebit_regex == None) or (not authbank_todebit_regex):
                logging.debug("\tauthbank_todebit_regex not successful; assigning authorizebank_regex[1] = {0}".format(authorizebank_regex[1]))
                fin_json["authorizebank"] = authorizebank_regex[1]
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
            #print(i)
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
            except_modify_text = i[i.upper().find('MODIFY') + 6:]
            logging.debug("\texcept_modify_text : {0}".format(except_modify_text))
            modify_check_box_text = ""
            if len(except_modify_text) > 2:
                modify_check_box_text = except_modify_text.split()[0]
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


        withbank_regex = re.search(r'[wi]th|n\s[B|G]ank(.*)', i, re.IGNORECASE)
        logging.debug("\twithbank_regex : {0}".format(withbank_regex))
        if withbank_regex:
            logging.debug("\twithbank_regex is successful, but it may contain ifsc also.")
        withbank_regex_alt = re.search(r'w.*[th|n]\s[B|G]ank\s(.*)', i, re.IGNORECASE)
        logging.debug("\twithbank_regex_alt : {0}".format(withbank_regex_alt))
        if withbank_regex_alt:
            logging.debug("\twithbank_regex_alt is successful, but it may contain ifsc also.")
        withbank_ifsc_regex = re.search(r'wi[th|n]\s[B|G]ank(.*)[ifs|fsc](.*)', i, re.IGNORECASE)
        logging.debug("\twithbank_ifsc_regex : {0}".format(withbank_ifsc_regex))
        if withbank_ifsc_regex:
            logging.debug("\twithbank_ifsc_regex is successful, meaning ifsc is in with withbank")
        if (withbank_regex and withbank_regex[1] != None ) and not withbank_ifsc_regex:
            logging.debug("\twithbank_regex is successful but withbank_ifsc_regex is not, means ifsc is not in with withbank.")
            logging.debug("\tchecking withbank_regex[1] : {0}".format(withbank_regex[1]))
            if len(withbank_regex[1]) >3:
                logging.debug("\tlen(withbank_regex[1]) >3 : {0}".format(withbank_regex[1]))
                fin_json['withbank'] = withbank_regex[1]
        elif withbank_regex_alt and not withbank_ifsc_regex:
            logging.debug("\twithbank_regex_alt is successful but withbank_ifsc_regex is not, means ifsc is not in with withbank.")
            if withbank_regex_alt[1] != None:
                if len(withbank_regex_alt[1]) >3:
                    logging.debug("\tlen(withbank_regex_alt[1]) >3 : {0}".format(withbank_regex_alt[1]))
                    fin_json['withbank'] = withbank_regex_alt[1]

        if (withbank_regex or withbank_regex_alt) and withbank_ifsc_regex:
            logging.debug("\twithbank_regex is successful, withbank_ifsc_regex is also successful; meaning ifsc is in with withbank.")
            logging.debug("\tin this case, possibility is that micr is also with withbank and ifsc in same line. so we introduce another regex for micr find.")
            withbank_ifsc_micr_regex = re.search(r'[with]\s[B|G]ank(.*)ifsc(.*)m.*cr(.*)', i, re.IGNORECASE)
            logging.debug("\twithbank_ifsc_micr_regex : {0}".format(withbank_ifsc_micr_regex))
            if withbank_ifsc_micr_regex:
                logging.debug("\twithbank_ifsc_micr_regex successful.")
                fin_json['ifsc'] = withbank_ifsc_micr_regex[2]
                logging.debug("\tfin_json['ifsc'] = withbank_ifsc_micr_regex[2] : {0}".format(withbank_ifsc_micr_regex[2]))
                fin_json['micr'] = removespace(withbank_ifsc_micr_regex[3])
                logging.debug("\tfin_json['micr'] = withbank_ifsc_micr_regex[3] : {0}".format(withbank_ifsc_micr_regex[3]))
            if not withbank_ifsc_micr_regex:
                logging.debug("\twithbank_ifsc_micr_regex is not successful, so we get ifsc from withbank_ifsc_regex. removespace(withbank_ifsc_regex[2]) : {0}".format(removespace(withbank_ifsc_regex[2])))
                fin_json['ifsc'] = removespace(withbank_ifsc_regex[2])
                if withbank_regex:
                    if "withbank" not in fin_json:
                        fin_json["withbank"] = withbank_regex[1]
                        logging.debug("\tfin_json['withbank'] = withbank_regex[1] : {0}".format(withbank_regex[1]))
                elif withbank_regex_alt:
                    if "withbank" not in fin_json:
                        fin_json["withbank"] = withbank_regex_alt[1]
                        logging.debug("\tfin_json['withbank'] = withbank_regex_alt[1] : {0}".format(withbank_regex_alt[1]))



        ifsc_micr_regex = re.search(r'.*fsc(.*)m.*cr(.*)', i, re.IGNORECASE)
        logging.debug("\tifsc_micr_regex : {0}".format(ifsc_micr_regex))
        if ifsc_micr_regex:
            logging.debug("\tifsc_micr_regex is successful.")
        if ifsc_micr_regex and not withbank_regex:
            logging.debug("\tifsc_micr_regex is successful and withbank_regex is not successful, means only ifsc and micr are in same line but not with withbank.")
            fin_json["ifsc"] = removespace(ifsc_micr_regex[1])
            logging.debug("\tfin_json['ifsc'] = ifsc_micr_regex[1] : {0}".format(ifsc_micr_regex[1]))
            fin_json['micr'] = removespace(ifsc_micr_regex[2])
            logging.debug("\tfin_json['micr'] = ifsc_micr_regex[2] : {0}".format(ifsc_micr_regex[2]))

        amountinwords_regex = re.search(r'.*amount\s.*rupees\s(.*)', i, re.IGNORECASE)
        logging.debug("\tamountinwords_regex : {0} is successful, meaning amountinwords is in the line, not sure about amount number though.".format(amountinwords_regex))
        amountinwords_amount_regex = re.search(r'.*amount\s.*rupees\s(.*)[\d{,9}]', i, re.IGNORECASE)
        logging.debug("\tamountinwords_amount_regex : {0} is successful, meaning amountinwords along with amount in number is present in same line.".format(amountinwords_amount_regex))
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

        frequency_check = "frequency" in i.lower()
        logging.debug("\tfrequency_check : {0}".format(frequency_check))
        frequency_regex = re.search(r'.*(?:freq.*y | f.*ency).*', i, re.IGNORECASE)
        logging.debug("\tfrequency_regex : {0}".format(frequency_regex))
        if frequency_regex or frequency_check:
            frequency_line_regex = re.search(r'(?:FR.*Y)(.*)(?:Mthy |M.*y)(.*)(?:Qtly |Q.*y)(.*)(?:H-yrly | H.*y)(.*)(?:Yrly | Y.*y)(.*)As.*', i, re.IGNORECASE)
            logging.debug("\tfrequency_line_regex : {0}".format(frequency_line_regex))
            if frequency_line_regex:
                if len(frequency_line_regex.group()) > 4:
                    monthly_tick_not_checked = "D" in frequency_line_regex[1].upper() or "Q" in frequency_line_regex[1].upper() or "O" in frequency_line_regex[1].upper() or "[]" in frequency_line_regex[1]
                    quarterly_tick_not_checked = "D" in frequency_line_regex[2].upper() or "Q" in frequency_line_regex[2].upper() or "O" in frequency_line_regex[2].upper() or "[]" in frequency_line_regex[2]
                    halfyearly_tick_not_checked = "D" in frequency_line_regex[3].upper() or "Q" in frequency_line_regex[3].upper() or "O" in frequency_line_regex[3].upper() or "[]" in frequency_line_regex[3]
                    yearly_tick_not_checked = "D" in frequency_line_regex[4].upper() or "Q" in frequency_line_regex[4].upper() or "O" in frequency_line_regex[4].upper() or "[]" in frequency_line_regex[4]
                    aawp_tick_not_checked = "D" in frequency_line_regex[5].upper() or "Q" in frequency_line_regex[5].upper() or "O" in frequency_line_regex[5].upper() or "[]" in frequency_line_regex[5]
                    if monthly_tick_not_checked == False:
                        fin_json["frequency"] = "Monthly"
                    if monthly_tick_not_checked == True:
                        if quarterly_tick_not_checked == False:
                            fin_json["frequency"] = "Quarterly"
                        if halfyearly_tick_not_checked == False:
                            fin_json["frequency"] = "Half-yearly"
                        if yearly_tick_not_checked == False:
                            fin_json["frequency"] = "Yearly"
                        if aawp_tick_not_checked == False:
                            fin_json["frequency"] = "As and when presented"




        reference1_regex = re.search(r'.*ref.*[1|?|\||L|l]\s(.*)', i, re.IGNORECASE)
        logging.debug("\treference1_regex first : {0}".format(reference1_regex))
        if not reference1_regex:
            reference1_regex = re.search(r'.*re[f|l].*\s[1|?|\||L|l]\s(.*)', i, re.IGNORECASE)
            logging.debug("\treference1_regex second : {0}".format(reference1_regex))
        if not reference1_regex:
            reference1_regex = re.search(r'.*r[en|ef|eh]ce.*[1|?|\||i|I|l]\s(.*)', i, re.IGNORECASE)
            logging.debug("\treference1_regex third : {0}".format(reference1_regex))
        reference1_mobile_regex = re.search(r'.*ref.*[1|?|\||L|l]\s(.*)mob.*[\d]{,11}', i, re.IGNORECASE)
        logging.debug("\treference1_mobile_regex : {0}".format(reference1_mobile_regex))
        if reference1_regex and not reference1_mobile_regex:
            logging.debug("\treference1_regex is successful but not reference1_mobile_regex; meaning only reference1 is in this line.")
            if "reference1" not in fin_json:
                fin_json["reference1"] = removespace(reference1_regex[1])
                logging.debug("\treference1 not in fin_json, removespace(reference1_regex[1]) : {0}".format(removespace(reference1_regex[1])))
            elif "reference1" in fin_json:
                logging.debug("\treference1 is in fin_json : {0}".format(fin_json["reference1"]))
                if (fin_json["reference1"] != None) or (fin_json["reference1"] != ""):
                    logging.debug("\tfin_json['reference1'] is not None or vacant.")
                    if len(fin_json["reference1"]) < 3:
                        logging.debug("\treference1 is in fin_json, but the length is < 3; appending : {0}".format(reference1_regex[1]))
                        fin_json["reference1"] = reference1_regex[1]
                elif fin_json["reference1"] == None:
                    logging.debug("\tfin_json['reference1'] is None; appending : {0}".format(removespace(reference1_regex[1])))
                    fin_json["reference1"] = removespace(reference1_regex[1])


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

        logging.debug("\tSearching for phone number for the scbl format.")
        phone_num_regex = re.search(r'.*ph.*n.*\s(.*)', i, re.IGNORECASE)
        logging.debug("\tphone_num_regex : {0}".format(phone_num_regex))
        if phone_num_regex:
            if "mobile" not in fin_json:
                fin_json["mobile"] = phone_num_regex[1]
                logging.debug("\tmobile not in fin_json and now phone_num_regex[1] is : {0}".format(phone_num_regex[1]))
            elif "mobile" in fin_json:
                if len(fin_json["mobile"]) <3:
                    logging.debug("\tmobile is in fin_json but its length is <3.")
                    fin_json["mobile"] = phone_num_regex[1]


        if "from" in i.lower():
            except_from_text = i[i.lower().find("from") + 4:]
            logging.debug("\texcept_from_text : {0}".format(except_from_text))
            check_alpha_char_in_fromdate = re.findall('[a-zA-Z]' , except_from_text)
            logging.debug("\tcheck_alpha_char_in_fromdate : {0}".format(check_alpha_char_in_fromdate))
            num_char_in_fromdate = re.findall('[0-9]' , except_from_text)
            logging.debug("\tnum_char_in_fromdate : {0}".format(num_char_in_fromdate))
            poss_fromdate = "".join(num_char_in_fromdate)
            if len(check_alpha_char_in_fromdate) < 3 and len(num_char_in_fromdate) > 3:
                if "fromdate" not in fin_json:
                    fin_json["fromdate"] = poss_fromdate
                elif "fromdate" in fin_json:
                    if len(fin_json["fromdate"]) < 3:
                        fin_json["fromdate"] = poss_fromdate
            elif len(check_alpha_char_in_fromdate) < 3 and len(num_char_in_fromdate) < 3:
                fromdate_next_line = sample_line_list[sample_line_list.index(i) +1]
                logging.debug("\tfromdate_next_line : {0}".format(fromdate_next_line))
                #poss_toline = sample_line_list[sample_line_list.index(i) + 2]
                poss_fromdate_nums = re.findall('[0-9]', fromdate_next_line)
                if len(poss_fromdate_nums) >3:
                    poss_fromdate_next = "".join(poss_fromdate_nums)
                    if "fromdate" not in fin_json:
                        fin_json["fromdate"] = poss_fromdate_next
                    elif "fromdate" in fin_json:
                        if len(fin_json["fromdate"]) < 3:
                            fin_json["fromdate"] = poss_fromdate_next

        elif "fro" in i.lower():
            except_fro_text = i[i.lower().find("fro") + 3:]
            logging.debug("\texcept_fro_text : {0}".format(except_fro_text))
            check_alpha_char_in_frodate = re.findall('[a-zA-Z]', except_fro_text)
            logging.debug("\tcheck_alpha_char_in_frodate : {0}".format(check_alpha_char_in_frodate))
            num_char_in_frodate = re.findall('[0-9]', except_fro_text)
            logging.debug("\tnum_char_in_frodate : {0}".format(num_char_in_frodate))
            poss_frodate = "".join(num_char_in_frodate)
            if len(check_alpha_char_in_frodate) < 4 and len(num_char_in_frodate) >3:
                if "fromdate" not in fin_json:
                    fin_json["fromdate"] = poss_frodate
                elif "fromdate" in fin_json:
                    if len(fin_json["fromdate"]) < 3:
                        fin_json["fromdate"] = poss_frodate


        if "to" in i.lower() and sample_line_list.index(i) > 6:
            except_to_text = i[i.lower().find("to") + 2:]
            logging.debug("\texcept_to_text : {0}".format(except_to_text))
            check_alpha_char_in_todate = re.findall('[a-zA-Z]' , except_to_text)
            logging.debug("\tcheck_alpha_char_in_todate : {0}".format(check_alpha_char_in_todate))
            num_char_in_todate = re.findall('[0-9]' , except_to_text)
            logging.debug("\tnum_char_in_todate : {0}".format(num_char_in_todate))
            poss_todate = "".join(num_char_in_todate)
            if len(check_alpha_char_in_todate) < 3 and len(num_char_in_todate) > 3:
                if "todate" not in fin_json:
                    fin_json["todate"] = poss_todate
                elif "todate" in fin_json:
                    if len(fin_json["todate"]) < 3:
                        fin_json["todate"] = poss_todate


        if "from" in i.lower() or "fron" in i.lower():
            logging.debug("\tfrom is in i.lower().")
            from_lineno = sample_line_list.index(i)
            logging.debug("\tfrom_lineno : {0}".format(from_lineno))
        until_cancel_regex_init = re.search(r'.*un.*can.*', i, re.IGNORECASE)
        logging.debug("\tuntil_cancel_regex_init : {0}".format(until_cancel_regex_init))
        if until_cancel_regex_init:
            poss_until_cancel_lineno = sample_line_list.index(i)
            logging.debug("\tposs_until_cancel_lineno : {0}".format(poss_until_cancel_lineno))
        if "from_lineno" in locals() and "poss_until_cancel_lineno" in locals():
            all_upper_lines = []
            logging.debug("\tfrom_lineno and poss_until_cancel_lineno are in locals.")
            for anyline in sample_line_list[from_lineno : poss_until_cancel_lineno]:
                logging.debug("\tline in between the from date and untilcancelled line : {0}".format(anyline))
                if anyline.isupper():
                    logging.debug("\tthe anyline is all uppercase : {0}".format(anyline))
                    all_upper_lines.append(anyline)
            if len(all_upper_lines) != 0:
                if "issigned1" not in fin_json:
                    fin_json["issigned1"] = all_upper_lines[-1]
                    logging.debug("\tas issigned1 is not in fin_json, anyline is getting appended into fin_json.")
                elif "issigned1" in fin_json:
                    if len(fin_json["issigned1"]) < 3:
                        logging.debug("\tissigned1 is is fin_json but the length of it is <3.")
                        fin_json["issigned1"] = all_upper_lines[-1]


        until_cancel_regex = re.search(r'.*unt.*ca.*', i, re.IGNORECASE)
        logging.debug("\tuntil_cancel_regex : {0}".format(until_cancel_regex))
        if not until_cancel_regex:
            until_cancel_regex = re.search(r'.*nti.*anc.*', i, re.IGNORECASE)
            logging.debug("\tuntil_cancel_regex second : {0}".format(until_cancel_regex))
        if until_cancel_regex:
            logging.debug("\tuntil_cancel_regex is successful.")
            until_cancel_line_no = sample_line_list.index(i)
            logging.debug("\tuntil_cancel_line_no : {0}".format(until_cancel_line_no))
            logging.debug("\tlen(sample_line_list) : {0}".format(len(sample_line_list)))
            until_cancel_next_line_no = until_cancel_line_no +1
            logging.debug("\tuntil_cancel_next_line_no : {0}".format(until_cancel_next_line_no))
            if until_cancel_next_line_no < len(sample_line_list):
                until_cancel_next_line = sample_line_list[until_cancel_next_line_no]
                logging.debug("\tuntil_cancel_next_line : {0}".format(until_cancel_next_line))
                if until_cancel_next_line.isupper():
                    if "issigned1" not in fin_json:
                        logging.debug("\tissigned1 not in fin_json.")
                        fin_json["issigned1"] = until_cancel_next_line
                    elif "issigned1" in fin_json:
                        if len(fin_json["issigned1"]) < 3:
                            logging.debug("\tissigned1 is in fin_json but mostly vacant.")
                            fin_json["issigned1"] = until_cancel_next_line


        """
        if "FREQ" in i.upper() or "frequency" in i.lower():
            keywords_pos["frequency"] = sample_line_list.index(i)
        reference2_regex = re.search(r'(.*)[ref](.*)[2](.*)', i, re.IGNORECASE)
        if reference2_regex:
            keywords_pos["reference2"] = sample_line_list.index(i)
    keywords_pos = sorted(keywords_pos.items(), key=lambda kv: kv[1])
    """
    logging.debug("\tBefore post-processing, fin_json : {0}".format(fin_json))
    logging.debug("\t\n\nNow post-processing on the fin_json starts.")

    logging.debug("\t\nPost_Process_1 : If debitaccounttype is not found in fin_json, we call our special function and assign if found.")
    if "debitaccounttype" not in fin_json:
        if find_todebit_line_only(sample_line_list):
            todebit_insert = find_todebit_line_only(sample_line_list)
            fin_json["debitaccounttype"] = todebit_insert
            logging.debug("\tPost_Process_1 : Completed.")

    logging.debug("\t\nPost_Process_1.1 : If reference1 is not found in fin_json, we call our special function and assign if found.")
    if "reference1" not in fin_json:
        if find_reference1_by_maxamount(sample_line_list):
            ref1_insert = find_reference1_by_maxamount(sample_line_list)
            fin_json["reference1"] = ref1_insert
            logging.debug("\tPost_Process_1.1 : Completed.")

    logging.debug("\t\nPost_Process_1.2 : If bankcode is not found in fin_json, we call our special function and assign if found.")
    if "bankcode" not in fin_json:
        if find_bankcode_from_nextline(sample_line_list):
            bankcode_insert = find_bankcode_from_nextline(sample_line_list)
            fin_json["bankcode"] = bankcode_insert
            logging.debug("\tPost_Process_1.2 : Completed.")
    
    logging.debug("\t\nPost_Process_1.3 : If bankcode is still not found in fin_json, we call our another special function and assign if found.")
    if "bankcode" not in fin_json:
        if find_bankcode_from_sponsorline(sample_line_list):
            bankcode_insert_2 = find_bankcode_from_sponsorline(sample_line_list)
            fin_json["bankcode"] = bankcode_insert_2
            logging.debug("\tPost_Process_1.3 : Completed.")

    logging.debug("\t\nPost_Process_1.4 : If authorizebank is not found in fin_json, we call our one special function and assign if dound.")
    if "authorizebank" not in fin_json:
        if find_authbank_by_strmatch(sample_line_list):
            authbank_insert = find_authbank_by_strmatch(sample_line_list)
            fin_json["authorizebank"] = authbank_insert
            logging.debug("\tPost_Process_1.4 : Completed.")
    elif "authorizebank" in fin_json:
        if len(fin_json["authorizebank"]) < 6 :
            if find_authbank_by_strmatch(sample_line_list):
                authbank_insert = find_authbank_by_strmatch(sample_line_list)
                fin_json["authorizebank"] = authbank_insert
                logging.debug("\tPost_Process_1.4 : Completed.")

    logging.debug("\t\nPost_Process_1.5 : If authorizebank is still not found in fin_json, we call our another special function and assign if found.")
    if "authorizebank" not in fin_json:
        if find_authorizebank_from_authoriseline(sample_line_list):
            authbank_insert2 = find_authorizebank_from_authoriseline(sample_line_list)
            fin_json["authorizebank"] = authbank_insert2
            logging.debug("\tPost_Process_1.5 : Completed.")
    elif "authorizebank" in fin_json:
        if len(fin_json["authorizebank"]) < 6:
            if find_authorizebank_from_authoriseline(sample_line_list):
                authbank_insert2 = find_authorizebank_from_authoriseline(sample_line_list)
                fin_json["authorizebank"] = authbank_insert2
                logging.debug("\tPost_Process_1.5 : Completed.")


    logging.debug("\t\nPost_Process_1.6 : If bankaccountnumber is not found in fin_json, we call our special function and assign if found.")
    if "bankaccountnumber" not in fin_json:
        if find_bank_acnumber(sample_line_list):
            acnum_insert = find_bank_acnumber(sample_line_list)
            logging.debug("\tPost_Process_1.6 : acnum_insert = {0}".format(acnum_insert))
            fin_json["acnumber"] = acnum_insert
            logging.debug("\tPost_Process_1.6 : Completed.")

    logging.debug("\t\nPost_Process_1.7 : If micr is not in fin_json, we call our special function and assign if found.")
    if "micr" not in fin_json:
        if find_micr_keyword(sample_line_list):
            micr_insert = find_micr_keyword(sample_line_list)
            fin_json["micr"] = micr_insert


    logging.debug("\t\nPost_Process_2 : If amount or amountinwords are not in fin_json, we call our special function and assign if found.")
    if "amount" not in fin_json and "amountinwords" not in fin_json:
        if find_amount(sample_line_list):
            amount_insert, amountinwords_insert = find_amount(sample_line_list)
            fin_json["amount"] = amount_insert
            fin_json["amountinwords"] = amountinwords_insert
            logging.debug("\tPost_Process_2 : Completed.")

    logging.debug("\t\nPost_Process_3 : If only amount is not in fin_json, still we call our special function and assign if found.")
    if "amount" not in fin_json and "amountinwords" in fin_json:
        if find_amount(sample_line_list):
            amount_insert_two, dummy_amountinwords = find_amount(sample_line_list)
            fin_json["amount"] = amount_insert_two
            logging.debug("\tPost_Process_3 : Completed.")

    logging.debug("\t\nPost_Process_3.1 : If amount is still not found in fin_json, we call our another special function and assign if found.")
    if "amount" not in fin_json:
        if find_amount_with_format(sample_line_list):
            amount_insert2 = find_amount_with_format(sample_line_list)
            fin_json["amount"] = amount_insert2
    elif "amount" in fin_json:
        if len(fin_json["amount"]) < 3:
            if find_amount_with_format(sample_line_list):
                amount_insert2 = find_amount_with_format(sample_line_list)
                fin_json["amount"] = amount_insert2

    logging.debug("\t\nPost_Process_4 : If we can't return any desired field, we fill them up with vacant strings.")
    for keyword in keywords_field_name:
        if keyword not in fin_json:
            logging.debug("\tPost_Process_4 : keyword not in fin_json is - {0}".format(keyword))
            fin_json[keyword] = ""

    logging.debug("\t\nPost_Process_5 : If we have any desired field as Nonetype, we change them to vacant strings.")
    for key in fin_json:
        if fin_json[key] == None:
            fin_json[key] = ""

    #logging.debug("\tfin_json before checking no of vacant values : {0}".format(fin_json))
    logging.debug("\t\nPost_Process_6 : Checking for the number of desired keys with vacant strings and assigning status of fin_json accordingly.")
    no_of_nonvacant_values = len([val for val in fin_json.values() if val != ""])
    logging.debug("\tPost_Process_6 : no_of_nonvacant_values = {0}".format(no_of_nonvacant_values))
    if no_of_nonvacant_values > 3:
        fin_json["status"] = "0"
    else:
        fin_json["status"] = "1"
    #print(fin_json)

    logging.debug("\t\nPost_Process_7 : If umrn is vacant, assigning nachtype as Create.")
    if fin_json["umrn"] == "":
        fin_json["nachtype"] = "CREATE"
        logging.debug("\tPost_Process_7 : Completed.")

    logging.debug("\t\nPost_Process_8 : If todate is vacant, assigning untilcancelled as True and debittype as Maximum Amount.")
    if fin_json["todate"] == "":
        fin_json["untilcancelled"] = "True"
        fin_json["debittype"] = "Maximum Amount"
        logging.debug("\tPost_Process_8 : Completed.")
    logging.debug("\t\nPost_Process_9 : removing unwanted special charcaters from withbank; returning only alphanumeric string in utilitycode, ifsc, micr.")
    fin_json["withbank"] = removespecialcharacter(fin_json["withbank"])
    fin_json["utilitycode"] = onlyalphanumeric(fin_json["utilitycode"])
    fin_json["ifsc"] = onlyalphanumeric(fin_json["ifsc"])
    fin_json["micr"] = onlyalphanumeric(fin_json["micr"])
    fin_json["amount"] = removespace(fin_json["amount"])

    logging.debug("\t\nPost_Process_10 : If ifsc length exceeds 11, take last four charcaters and check if any alphabetic charcaters are attached.")
    if len(fin_json["ifsc"]) > 11:
        fin_json["ifsc"] = fin_json["ifsc"][:-4] + "".join([i for i in fin_json["ifsc"][-4:] if i.isnumeric()])
        logging.debug("\tPost_Process_10 : Completed.")

    logging.debug("\t\nPost_Process_11 : If bankcode contains 'HSBC', check the immediate next charcater for 'O' and if present replace with '0'.")
    if "HSBC" in fin_json["bankcode"]:
        if fin_json["bankcode"][fin_json["bankcode"].find("HSBC") + 4] == "O":
            fin_json["bankcode"] = fin_json["bankcode"][: fin_json["bankcode"].find("HSBC") + 4] + "0" + fin_json["bankcode"][fin_json["bankcode"].find("HSBC") + 5:]
            logging.debug("\tPost_Process_11 : Completed.")

    logging.debug("\t\nPost_Process_12 : If length of bankcode exceeds 4 and if the fourth charcater is 'O', replace it with '0'.")
    if len(fin_json["bankcode"]) > 4:
        if fin_json["bankcode"][4] == "O":
            fin_json["bankcode"] = fin_json["bankcode"][:4] + "0" + fin_json["bankcode"][5:]
            logging.debug("\tPost_Process_12 : Completed.")

    logging.debug("\t\nPost_Process_13 : If length of authorizebank exceeds 10, check if any charcater associated with to debit are associated, if found remove them.")
    if len(fin_json["authorizebank"]) > 10:
        if " to" in fin_json["authorizebank"].lower():
            fin_json["authorizebank"] = fin_json["authorizebank"][:fin_json["authorizebank"].lower().find(" to")]
        elif "to " in fin_json["authorizebank"].lower():
            fin_json["authorizebank"] = fin_json["authorizebank"][:fin_json["authorizebank"].lower().find("to ")]
        if "deb" in fin_json["authorizebank"].lower():
            fin_json["authorizebank"] = fin_json["authorizebank"][:fin_json["authorizebank"].lower().find("deb")]

    logging.debug("\t\nPost_Process_14 : If micr contains only one charcater, regardless of being just once or duplicated; it's ought to be wrong. Reassign micr to vacant string.")
    if len(set(fin_json["micr"])) == 1:
        if "I" in set(fin_json["micr"]):
            fin_json["micr"] = ""
            logging.debug("\tPost_Process_14 : Completed.")

    logging.debug("\t\nPost_Process_15 : If authorizebank is not vacant, find if 'LIM' is in it; if present, force reassign it to 'LIMITED'.")
    if fin_json["authorizebank"] != "":
        authorizebank_split = fin_json["authorizebank"].split()
        if "LIM" in authorizebank_split[-1].upper():
            authorizebank_split = authorizebank_split[:-1] + ["LIMITED"]
            authorizebank_joint = " ".join(authorizebank_split)
            fin_json["authorizebank"] = authorizebank_joint
            logging.debug("\tPost_Process_15 : Completed.")

    logging.debug("\t\nPost_Process_16 : If length of ifsc concedes 5, check if fourth charcater is 'D' or 'Q' or 'O'; if so, replace them with '0'.")
    if len(fin_json["ifsc"]) > 5:
        ifsc_o_char = fin_json["ifsc"][4]
        logging.debug("\tPost_Process_16 : ifsc_o_char = {0}".format(ifsc_o_char))
        if ifsc_o_char == "D" or ifsc_o_char == "O" or ifsc_o_char == "Q":
            fin_json["ifsc"] = fin_json["ifsc"][:4] + "0" + fin_json["ifsc"][5:]
            logging.debug("\tPost_Process_16 : Completed.")

    logging.debug("\t\nPost_Process_17 : If length of bankcode exceeds 6, check if 'HSBC02' in bankcode; if yes force reassign bankcode to 'HSBC02INDIA'.")
    if len(fin_json["bankcode"]) > 6:
        if "HSBC02" in fin_json["bankcode"]:
            fin_json["bankcode"] = "HSBC02INDIA"
            logging.debug("\tPost_Process_17 : Completed.")
    """
    logging.debug("\t\nPost_Process_18 : If length of ifsc code exceeds 4, check for possible wrongly read charcaters and replace them with precognitive words.")
    if len(fin_json["ifsc"]) > 4:
        if "C1T1" in fin_json["ifsc"]:
            fin_json["ifsc"] = fin_json["ifsc"].replace("C1T1", "CITI")
        if "C1T" in fin_json["ifsc"]:
            fin_json["ifsc"] = fin_json["ifsc"].replace("C1T", "CIT")
        if "CIT1" in fin_json["ifsc"]:
            fin_json["ifsc"] = fin_json["ifsc"].replace("CIT1", "CITI")
        if "TOBA" in fin_json["ifsc"]:
            fin_json["ifsc"] = fin_json["ifsc"].replace("TOBA", "IOBA")
        if "HDEC" in fin_json["ifsc"]:
            fin_json["ifsc"] = fin_json["ifsc"].replace("HDEC", "HDFC")
        if "1C1C" in fin_json["ifsc"]:
            fin_json["ifsc"] = fin_json["ifsc"].replace("1C1C", "ICIC")
        if "1C1" in fin_json["ifsc"]:
            fin_json["ifsc"] = fin_json["ifsc"].replace("1C1", "ICI")
    """
    logging.debug("\t\nPost_Process_18 : If length of ifsc code exceeds 4, check for possible wrongly read charcaters and replace them with precognitive words from lookup.")
    if len(fin_json["ifsc"]) > 4:
        ifsc_first_four_char = fin_json["ifsc"][:4]
        for k,v in ifsc_mod_dict.items():
            if ifsc_first_four_char in v:
                fin_json["ifsc"] = k + fin_json["ifsc"][4:]

    logging.debug("\t\nPost_Process_19 : If length of ifsc exceeds 8, find for 'O' or 'D' or 'Q' after fourth items and replace them with '0'.")
    if len(fin_json["ifsc"]) > 8:
        ifsc_alpha_char = fin_json["ifsc"][:4]
        ifsc_num_char = fin_json["ifsc"][4:]
        fin_json["ifsc"] = ifsc_alpha_char + ifsc_num_char.replace("O", "0").replace("Q", "0").replace("o", "0").replace("D", "0")
        logging.debug("\tPost_Process_19 : Completed.")

    logging.debug("\t\nPost_Process_20 : If length of utilitycode exceeds 8, find for 'O' or 'D' or 'Q' after fourth items and replace them with '0'.")
    if len(fin_json["utilitycode"]) > 8:
        fin_json["utilitycode"] = fin_json["utilitycode"][:5] + fin_json["utilitycode"][5:].replace("O" , "0").replace("Q" , "0").replace("o", "0")
        logging.debug("\tPost_Process_20 : Completed.")

    logging.debug("\t\nPost_Process_21 : If length of bankcode exceeds 4, check for possible wrongly read charcaters and replace them with precognitive words.")
    if len(fin_json["bankcode"]) >4:
        if "1ND1A" in fin_json["bankcode"]:
            fin_json["bankcode"] = fin_json["bankcode"].replace("1ND1A", "INDIA")
        if "1NDIA" in fin_json["bankcode"]:
            fin_json["bankcode"] = fin_json["bankcode"].replace("1NDIA", "INDIA")
        if "IND1A" in fin_json["bankcode"]:
            fin_json["bankcode"] = fin_json["bankcode"].replace("IND1A", "INDIA")
        if "1ND" in fin_json["bankcode"]:
            fin_json["bankcode"] = fin_json["bankcode"].replace("1ND", "IND")
        if "NDUA" in fin_json["bankcode"]:
            fin_json["bankcode"] = fin_json["bankcode"].replace("NDUA", "NDIA")

    logging.debug("\t\nPost_Process_22 : If length of ifsc exceeds 4, check if withbank is vacant and change withbank with help of ifsc:withbank lookup dictionary.")
    if len(fin_json["ifsc"]) > 4:
        """if "HDFC" in fin_json["ifsc"]:
                if len(fin_json["withbank"]) < 3:
                    fin_json["withbank"] = "HDFC Bank"
        """
        for eachpair in ifsc_withbank_dict:
            if eachpair in fin_json["ifsc"]:
                if len(fin_json["withbank"]) < 3:
                    fin_json["withbank"] = ifsc_withbank_dict[eachpair]

    logging.debug("\t\nPost_Process_23 : If frequency is vacant, force assign 'Monthly' to it.")
    if fin_json["frequency"] == "":
        fin_json["frequency"] = "Monthly"

    logging.debug("\t\nPost_Process_24 : If micr contains 'G', replace it with '6'.")
    if len(fin_json["micr"]) > 2:
        fin_json["micr"] = fin_json["micr"].replace("G", "6")
    
    logging.debug("\t\nPost_Process_25 : If 'rupees' similar word is found in amountinwords then remove the part upto that word.")
    if len(fin_json["amountinwords"]) > 10 :
        if "ees" in fin_json["amountinwords"].lower():
            fin_json["amountinwords"] = fin_json["amountinwords"][fin_json["amountinwords"].lower().find("ees") + 3 :]

    logging.debug("\t\nPost_Process_26 : Remove all charcaters except the alphabetic ones from amountinwords.")
    fin_json["amountinwords"] = onlyalphabetic(fin_json["amountinwords"])

    logging.debug("\t\nPost_Process_26 : If amount is present in fin_json but amountinwords is vacant, call our special function to assign amountinwords.")
    if len(fin_json["amountinwords"]) < 3 and len(fin_json["amount"]) >1:
        fin_json["amountinwords"] = num2word(fin_json["amount"])
        logging.debug("\tPost_Process_26 : Completed.")

    logging.debug("\t\nPost_Process_27 : If amountinwords is present in fin_json but amount is vacant, call our special function to assign amount.")
    if len(fin_json["amountinwords"]) > 3 and len(fin_json["amount"]) < 2:
        fin_json["amount"] = str(word2num(fin_json["amountinwords"]))
        logging.debug("\tPost_Process_27 : Completed.")

    logging.debug("\t\nPost_Process_28 : If debitaccounttype in fin_json has unwanted charcaters, remove them after finding our keywords.")
    if len(fin_json["debitaccounttype"]) > 8:
        if "SB" in fin_json["debitaccounttype"].upper() and "NRE" not in fin_json["debitaccounttype"].upper() and "NRO" not in fin_json["debitaccounttype"].upper():
            fin_json["debitaccounttype"] = "SB"
        if "SB" in fin_json["debitaccounttype"].upper() and "NRE" in fin_json["debitaccounttype"].upper():
            fin_json["debitaccounttype"] = "SB-NRE"
        if "SB" in fin_json["debitaccounttype"].upper() and "NRO" in fin_json["debitaccounttype"].upper():
            fin_json["debitaccounttype"] = "SB-NRO"
        if "CA " in fin_json["debitaccounttype"].upper() or " CA" in fin_json["debitaccounttype"].upper():
            fin_json["debitaccounttype"] = "CA"
        if "CC " in fin_json["debitaccounttype"].upper() or " CC" in fin_json["debitaccounttype"].upper():
            fin_json["debitaccounttype"] = "CC"

    logging.debug("\t\nPost_Process_29 : If immediate charcater after 'NACH' in utilitycode is 'O', change it to '0'.")
    if len(fin_json["utilitycode"]) > 4:
        if "NACH" in fin_json["utilitycode"].upper():
            if fin_json["utilitycode"][fin_json["utilitycode"].upper().find("NACH")+4] == "O":
                fin_json["utilitycode"] = fin_json["utilitycode"][:fin_json["utilitycode"].upper().find("NACH")+4] + "0" + fin_json["utilitycode"][fin_json["utilitycode"].upper().find("NACH")+5:]
                logging.debug("\tPost_Process_29 : Completed.")

    logging.debug("\t\nPost_Process_30 : If debitaccounttype is vacant, assign it to SB.")
    if len(fin_json["debitaccounttype"]) < 2:
        fin_json["debitaccounttype"] = "SB"
    if fin_json["debitaccounttype"] == "":
        fin_json["debitaccounttype"] = "SB"
        logging.debug("\tPost_Process_30 : Completed.")

    logging.debug("\t\nPost_Process_31 : If todate is not vacant, assign untilcancelled as False.")
    if len(fin_json["todate"]) > 2:
        fin_json["untilcancelled"] = "False"
        logging.debug("\tPost_Process_31 : Completed.")

    logging.debug("\t\nPost_Process_33 : If todate is not in proper format ie extra 1s are read, remove single occurance of them after last '20' found and replace multiple occurance by a single one.")
    if fin_json["todate"] != "":
        digits_after_last20 = fin_json["todate"][fin_json["todate"].rfind("20")+2:]
        digits_before_last20 = fin_json["todate"][:fin_json["todate"].rfind("20")]
        if len(digits_after_last20) >2:
            digits_after_last20 = re.sub(r'1{2,}', '1', re.sub(r'(?<!1)1(?=[^1]|$)', '', digits_after_last20))
            #digits_after_last20 = digits_after_last20.replace("1","")
            fin_json["todate"] = fin_json["todate"][:fin_json["todate"].rfind("20")+2] + digits_after_last20
        if len(digits_before_last20) == 3:
            fin_json["todate"] = '0' + fin_json["todate"]

    logging.debug("\t\nPost_Process_33 : If fromdate is not in proper format ie extra 1s are read, remove single occurance of them after last '20' found and replace multiple occurance by a single one.")
    if fin_json["fromdate"] != "":
        if fin_json["fromdate"].rfind("20") != -1:
            fromdate_digits_after_last20 = fin_json["fromdate"][fin_json["fromdate"].rfind("20")+2:]
            fromdate_digits_before_last20 = fin_json["fromdate"][:fin_json["fromdate"].rfind("20")]
            if len(fromdate_digits_after_last20) >2:
                fromdate_digits_after_last20 = re.sub(r'1{2,}', '1', re.sub(r'(?<!1)1(?=[^1]|$)', '', fromdate_digits_after_last20))
                #digits_after_last20 = digits_after_last20.replace("1","")
                fin_json["fromdate"] = fin_json["fromdate"][:fin_json["fromdate"].rfind("20")+2] + fromdate_digits_after_last20
            if len(fromdate_digits_before_last20) == 3:
                fin_json["fromdate"] = '0' + fin_json["fromdate"]

    logging.debug("\t\nPost_Process_34 : In ifsc if total number of digits is not 7, check first occurance of zero and omit the previous digits.")
    if len([k for k in fin_json["ifsc"] if k.isnumeric()]) > 7:
        if len(fin_json["ifsc"][ fin_json["ifsc"].find('0') :]) == 7:
            fin_json["ifsc"] = "".join([k for k in fin_json["ifsc"] if not k.isnumeric()]) + fin_json["ifsc"][ fin_json["ifsc"].find('0') :]

    logging.debug("\t\nPost_Process_35 : If debittype is vacant, assign Maximum Amount.")
    if fin_json["debittype"] == "":
        fin_json["debittype"] = 'Maximum Amount'

    logging.debug("\t\nPost_Process_36 : If length of debitaccounttype is very high, assign to 'SB'.")
    if len(fin_json["debitaccounttype"]) > 9:
        fin_json["debitaccounttype"] = 'SB'

    logging.debug("\t\nPost_Process_37 : If withbank is not found, call our special function and if found assign.")
    if fin_json["withbank"] == "":
        if find_withbank(sample_line_list):
            withbank_insert = find_withbank(sample_line_list)
            fin_json["withbank"] = withbank_insert

    logging.debug("\t\nPost_Process_38 : If withbank contains special charcaters, remove them by applying function onlyalphabetic.")
    fin_json["withbank"] = onlyalphabetic(fin_json["withbank"])

    logging.debug("\t\nPost_Process_39 : If number of unique characters in micr is less than 3, it may contain arbitrary charcaters.")
    if len(list(set(fin_json["micr"].lower()))) <= 3:
        fin_json["micr"] = ""

    logging.debug("\t\nPost_Process_40 : If withbank is read, check if the alphabetic charcaters match with the cooresponding bank code.")
    if fin_json["withbank"] != "" and fin_json["ifsc"] != "":
        withbank_to_check_in_dict = " ".join(fin_json["withbank"].lower().split())
        ifsc_alphabetic_chars = "".join([m for m in fin_json["ifsc"] if not m.isnumeric()])
        ifsc_num_chars = "".join([m for m in fin_json["ifsc"] if m.isnumeric()])
        if withbank_to_check_in_dict in withbank_ifsc_dict:
            if withbank_ifsc_dict[withbank_to_check_in_dict] != ifsc_alphabetic_chars:
                fin_json["ifsc"] = withbank_ifsc_dict[withbank_to_check_in_dict] + ifsc_num_chars
                logging.debug("\tPost_Process_40 : Completed.")

    logging.debug("\t\nPost_Process_41 : If withbank is found, but ifsc is not read; call our special function to check for ifsc and assign if found.")
    if len(fin_json["withbank"]) > 4 and len(fin_json["ifsc"]) < 4:
        withbank_clean = " ".join(fin_json["withbank"].lower().split())
        _, withbank_sim_key_tolook = check_string_similarity_list(withbank_clean, withbank_ifsc_dict.keys())
        logging.debug("\tPost_Process_41 : withbank_sim_key_tolook = {0}".format(withbank_sim_key_tolook))
        if find_ifsc_if_withbank_found(sample_line_list, withbank_ifsc_dict[withbank_sim_key_tolook]):
            ifsc_insert = find_ifsc_if_withbank_found(sample_line_list, withbank_ifsc_dict[withbank_sim_key_tolook])
            logging.debug("\tPost_Process_41 : ifsc_insert = {0}".format(ifsc_insert))
            fin_json["ifsc"] = ifsc_insert
            logging.debug("\tPost_Process_41 : Completed.")

    logging.debug("\t\nPreparaing a new json for maintaining the correct order of keys.")
    new_js = {}
    for key in keywords_field_name:
        new_js[key] = fin_json[key]
    pprint(new_js)
    logging.debug("\tnew_js : {0}".format(new_js))
    return new_js

def final_call(file_name_fc):
    final_js = ret_json(get_line_lists(file_name_fc))
    #name = os.path.splitext(file_name)[0]
    name = file_name_fc.split("/")[-1].split(".")[0]
    json_filename = name + ".json"
    with open(json_filename, "w") as jsonoutfile:
        json.dump(final_js, jsonoutfile, indent = 4)
    logging.debug("\tThe json file is successfully created.")
if check_rot == True:
    final_call(output_img_filename)
else:
    final_call(file_name)