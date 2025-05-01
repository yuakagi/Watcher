import seaborn as sns

# Colors for main background
bg_tone = ["#f7f5ee", "#e3e0cd", "#c7c2a6", "#a8a486", "#898265", "#625f4a"]
bg_gray = ["#e6e6ed", "#c8cdda", "#99a3b4", "#6f7c91", "#49566d", "#243348"]

# Colors for main accents
ac_red = ["#fad0ce", "#eca4a6", "#de6866", "#ca3f3f", "#9b2726", "#751915"]
ac_blue = ["#c8e7fb", "#9ccbec", "#5799d2", "#0072b2", "#044f91", "#032c5d"]
ac_yellow = ["#fff1c3", "#f6dc8a", "#ebc851", "#cba02b", "#9d7717", "#6b5514"]

# Colors for extended palettes
ext_olive = ["#f3f1b3", "#e0dc67", "#cac700", "#9aa415", "#65771d", "#36451a"]
ext_green = ["#dae8c6", "#a0ca79", "#62b346", "#469433", "#24722f", "#143c1c"]
ext_teal = ["#cce7ee", "#98d2d4", "#4abcbd", "#009aa3", "#036879", "#033950"]
ext_blue = ["#c9e7fc", "#9dccec", "#5799d2", "#0172b3", "#004e91", "#032b5c"]
ext_purple = ["#ead6e9", "#d3a9d1", "#bb7cb4", "#a84e94", "#792c74", "#481951"]
ext_red = ["#f9cfce", "#eda4a7", "#de6866", "#c43d3d", "#9b2726", "#751915"]
ext_orange = ["#fde0bd", "#fbc07f", "#f49945", "#eb6d00", "#b65008", "#833200"]
ext_yellow = ["#fff2c3", "#f6dc8a", "#eac750", "#cc9f2b", "#9d7818", "#6b5614"]
ext_skin_tone = ["#f9e6d7", "#ddbea2", "#bf997d", "#946b57", "#765041", "#452d20"]

# Rotations
rotation_1 = [
    "#4abcbd",  # Teal
    "#bb7cb4",  # Purple
    "#f49945",  # Orange
    "#62b346",  # Green
    "#5799d2",  # Blue
    "#de6866",  # Red
    "#cac700",  # Olive
]

CUSTOM_PALETTES = {
    "bg_tone": sns.color_palette(bg_tone),
    "bg_gray": sns.color_palette(bg_gray),
    "ac_red": sns.color_palette(ac_red),
    "ac_blue": sns.color_palette(ac_blue),
    "ac_yellow": sns.color_palette(ac_yellow),
    "ext_olive": sns.color_palette(ext_olive),
    "ext_green": sns.color_palette(ext_green),
    "ext_teal": sns.color_palette(ext_teal),
    "ext_blue": sns.color_palette(ext_blue),
    "ext_purple": sns.color_palette(ext_purple),
    "ext_red": sns.color_palette(ext_red),
    "ext_orange": sns.color_palette(ext_orange),
    "ext_yellow": sns.color_palette(ext_yellow),
    "ext_skin_tone": sns.color_palette(ext_skin_tone),
    "rotation_1": sns.color_palette(rotation_1),
}
