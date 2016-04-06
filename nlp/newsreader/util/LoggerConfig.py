import logging

logger_config = {
    "level"     : logging.INFO,
    "format"    : '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    "datefmt"   : '%m-%d %H:%M',
    "filename"  : 'newsreader.log',
    "filemode"  : 'w'
}
