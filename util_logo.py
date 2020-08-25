""" Usage

fname = 'toto.png'
fnameB = 'toto_final.png'
fig = fct_to_create_plot(ds)
fig.savefig(fname, orientation='landscape', dpi=120, bbox_inches='tight')
plt.close(fig)
add_lowerband(fname, fnameB, band_height=80)
add_2logo(fnameB, fnameB, logo_height=80, data_src='Mercator-blabla')
            
"""
def add_2logo(mfname, outfname, logo_height=120, txt_color=(0, 0, 0, 255), data_src='CMEMS'):
    """ Add 2 logos and text to a figure

        Parameters
        ----------
        mfname : string
            source figure file
        outfname : string
            output figure file
    """

    # lfname = "misc/ArgoFR_Logo_300.png"
    lfname2 = "misc/logo-lops-big2.png"
    lfname1 = "misc/logo_lefe.png"

    mimage = Image.open(mfname)

    # Open logo images:
    limage1 = Image.open(lfname1)
    limage2 = Image.open(lfname2)

    # Resize logos to match the requested logo_height:
    aspect_ratio = limage1.size[1]/limage1.size[0] # height/width
    simage1 = limage1.resize((int(logo_height/aspect_ratio), logo_height) )

    aspect_ratio = limage2.size[1]/limage2.size[0] # height/width
    simage2 = limage2.resize((int(logo_height/aspect_ratio), logo_height) )

    # Paste logos along the lower white band of the main figure:
    box = (0, mimage.size[1]-logo_height)
    mimage.paste(simage1, box)

    box = (simage1.size[0], mimage.size[1]-logo_height)
    mimage.paste(simage2, box)

    # Add copyright text:
    txtA = ("© 2017-2019, SOMOVAR Project, %s") % (__author__)
    fontA = ImageFont.truetype(font_path, 20)

    txtB = "Data source: %s" % data_src
    fontB = ImageFont.truetype(font_path, 16)

    txtsA = fontA.getsize_multiline(txtA)
    txtsB = fontB.getsize_multiline(txtB)

    xoffset = 5 + simage1.size[0] + simage2.size[0]
    if 0:  # Align text to the top of the band:
        posA = (xoffset, mimage.size[1]-logo_height - 1)
        posB = (xoffset, mimage.size[1]-logo_height + txtsA[1])
    else:  # Align text to the bottom of the band:
        posA = (xoffset, mimage.size[1]-txtsA[1]-txtsB[1]-5)
        posB = (xoffset, mimage.size[1]-txtsB[1]-5)

    # Print
    drawA = ImageDraw.Draw(mimage)
    drawA.text(posA, txtA, txt_color, font=fontA)
    drawB = ImageDraw.Draw(mimage)
    drawB.text(posB, txtB, txt_color, font=fontB)

    # Final save
    mimage.save(outfname)


def add_lowerband(mfname, outfname, band_height = 120, color=(255, 255, 255, 255)):
    image = Image.open(mfname, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]
    background = Image.new('RGBA', (width, height + band_height), color)
    background.paste(image, (0, 0))
    background.save(outfname)
