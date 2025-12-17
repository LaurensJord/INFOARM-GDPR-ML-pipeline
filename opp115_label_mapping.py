"""
0                Other      8
2         PolicyChange      5
3       DataCollection      0
10       DataRetention      3
20    SpecialAudiences      7
27   ThirdPartySharing      1
29          UserRights      2
103       DataSecurity      4
110         DoNotTrack      6
"""

# OPP_115 Label Mapping
label_names_list = ["Other", "PolicyChange", "DataCollection", "DataRetention", "SpecialAudiences", "ThirdPartySharing", "UserRights", "DataSecurity", "DoNotTrack"]
labels_list = [8, 5, 0, 3, 7, 1, 2, 4, 6]

# OPP_115 mapping to GDPR Principles 
OPP10_TO_GDPR = {
    "First Party Collection/Use": ["Lawfulness, fairness and transparency", "Purpose limitation", "Data minimization"],
    "Third Party Sharing/Collection": ["Lawfulness, fairness and transparency", "Purpose limitation", "Data minimization"],
    "User Choice/Control": ["Lawfulness, fairness and transparency"],
    "User Access, Edit, and Deletion": ["Lawfulness, fairness and transparency", "Accuracy"],
    "Data Retention": ["Storage limitation"],
    "Data Security": ["Integrity and confidentiality"],
    "Policy Change": ["Lawfulness, fairness and transparency"],
    "Do Not Track": [],
    "Intl. and Specific Audiences": ["Accountability"],
    "Other": [],
}

OPP9_TO_GDPR = {
    "DataCollection": OPP10_TO_GDPR["First Party Collection/Use"],
    "ThirdPartySharing": OPP10_TO_GDPR["Third Party Sharing/Collection"],
    # UserRights = union van beide canonical user-categorieën
    "UserRights": sorted(set(
        OPP10_TO_GDPR["User Choice/Control"] +
        OPP10_TO_GDPR["User Access, Edit, and Deletion"]
    )),
    "DataRetention": OPP10_TO_GDPR["Data Retention"],
    "DataSecurity": OPP10_TO_GDPR["Data Security"],
    "PolicyChange": OPP10_TO_GDPR["Policy Change"],
    "DoNotTrack": OPP10_TO_GDPR["Do Not Track"],
    "SpecialAudiences": OPP10_TO_GDPR["Intl. and Specific Audiences"],
    "Other": OPP10_TO_GDPR["Other"],
}

def gdpr_principles_for_opp115_label(label_name):
    return OPP9_TO_GDPR.get(label_name, [])