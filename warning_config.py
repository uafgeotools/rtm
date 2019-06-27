import warnings


# Subclass UserWarning
class RTMWarning(UserWarning):
    pass


# Make a custom warning format for RTMWarning instances
def _rtm_warning_format(message, category, *args, **kwargs):
    if category == RTMWarning:
        msg = f'{category.__name__}: {message}\n'  # Much cleaner
    else:
        msg = warnings.WarningMessage(message, category, *args, **kwargs)
        msg = warnings._formatwarnmsg_impl(msg)  # Default
    return msg


# Make warnings cleaner and more consistent
warnings.formatwarning = _rtm_warning_format
warnings.filterwarnings(category=RTMWarning, action='always')
