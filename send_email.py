"""
author: Florian Krach
"""

import smtplib
from os.path import basename
from email.mime.application import MIMEApplication
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.Utils import formatdate


def send_email(mailadress='', smtpserver='', port=587, username='',
               password='', to='',
               subject='python mail', body="DONE", file_names=None):
    '''
    :param mailadress:
    :param smtpserver:
    :param port:
    :param username:
    :param password:
    :param to:
    :param subject:
    :param body:
    :param file_names: either None or a list containing the filenames as strings
    :return:
    '''

    From = mailadress
    msg = MIMEMultipart()
    msg['From'] = From
    msg['To'] = to
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # attach files:
    if file_names is not None:
        for f in file_names:
            with open(f, "rb") as fil:
                part = MIMEApplication(fil.read(), Name=basename(f))
            # After the file is closed
            part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
            msg.attach(MIMEText('\n\n', 'plain'))
            msg.attach(part)

    text = msg.as_string()

    server = smtplib.SMTP(smtpserver, port)
    server.ehlo()  # Has something to do with sending information
    server.starttls()  # Use encrypted SSL mode
    server.ehlo()  # To make starttls work
    server.login(username, password)
    failed = server.sendmail(From, to, text)
    server.quit()

    return failed

# print send_email()

