import logging, sys
import newspaper
import simplejson as json

from util               import LoggerConfig
from newspaper          import Config, Source
from dao                import MongoDao
from concurrent.futures import ThreadPoolExecutor, wait, as_completed

logging.basicConfig(**LoggerConfig.logger_config)
logger = logging.getLogger(__name__)

newsDao = MongoDao.NewsArticleDao()

class NewsCrawlerConfig(object):
    SITE_FILE  = "config/news_sites.json"
    CRAWL_FILE = "config/crawl_options.json" 
    class NewsSite(object):
        def __init__(self, name, url, crawl_threads):
            self.name = name
            self.url  = url
            self.crawl_threads = crawl_threads

    def __init__(self):
        self.sites        = []
        self.crawl_option = Config() 
        self.is_config_read = False
        
    def as_newscrawler(self, site_obj):
        self.sites.append(NewsCrawlerConfig.NewsSite(site_obj['name'],
                                                     site_obj['url'], site_obj['threads']))
        return None

    def as_crawloptions(self, crawl_obj):
        self.crawl_option.__dict__.update(crawl_obj)
        return None
        
    def read_config(self):
        try:
            logger.info("going to read the sites config file ")
            with open(NewsCrawlerConfig.SITE_FILE, "rbU") as config:
                try:
                    logger.info("going to json load the sites config")
                    json.load(config, object_hook = self.as_newscrawler)
                    self.is_config_read = True
                except Exception as load_e:
                    logger.error(load_e)
                    raise load_e
        except IOError as file_e:
            logger.error(file_e)
            raise file_e

        try:
            logger.info("going to read the crawler configurations")
            with open(NewsCrawlerConfig.CRAWL_FILE, "rbU") as config:
                try:
                    logger.info("going to json load the crawler configs")
                    json.load(config, object_hook = self.as_crawloptions)
                except Exception as load_e:
                    logger.error(load_e)
                    raise load_e
        except IOError as file_e:
            logger.error(file_e)
            raise file_e
        
    def is_read(self):
        return self.is_config_read
        
class NewsCrawler(object):

    def __init__(self, config):
        assert config.__class__.__name__ == "NewsCrawlerConfig", \
            """config must be an instance of class NewsCrawlerConfig"""
        self.crawler_config = config

    def article_callback(self, f_obj):
        logger.info("Article formulation callback on future object %s", str(f_obj))
        try:
            article    = f_obj.result()
            a_source   = article.source_url
            a_title    = article.title
            a_location = article.url
            a_text     = article.text
            a_summary  = article.summary
            a_date     = article.publish_date
            a_meta     = article.meta_data
            a_keywords, a_open_graph = None, None
            if 'keywords' in a_meta:
                a_keywords = a_meta['keywords'].split(",")
            if 'og' in a_meta:
                a_open_graph = a_meta['og']
            news_article = newsDao.form(a_source, a_title, a_location, a_text,
                                        a_summary, a_keywords, a_open_graph,
                                        a_date)
            newsDao.save(news_article)
        except Exception, e:
            logger.error(e)

    def __sanitize_text(self, text):
        clean_text = "".join((c for c in text if 0 < ord(c) < 127))
        clean_text = clean_text.replace("\n", " ").replace("\t", " ")
        return clean_text
        
    def download_article(self, article):
        try:
            logger.info("will download the article from source")
            article.download()
            if self.do_parse:
                logger.info("Will parse the document")
                article.parse()
                
                article.set_text(self.__sanitize_text(article.text))
                logger.info("parsed and formed the document text")
                
                article.set_title(self.__sanitize_text(article.title))
                logger.info("parsed and formed the document title")

            if self.do_nlp  :
                article.nlp()
                article.set_summary(self.__sanitize_text(article.summary))
                logger.info("preformed nlp and formed the document summary")

            return article

        except Exception as e:
            logger.error(e)
            raise e

    def __init_crawl(func):
        def wrapped(self, *args, **kwargs):
            if not self.crawler_config.is_read(): self.crawler_config.read_config()
            workers = self.crawler_config.crawl_option.number_threads
            self.news_pools = {}
            logger.info("Will form the pools for the newspaper sources")
            for source in self.crawler_config.sites:
                self.news_pools[source.name] = ThreadPoolExecutor(max_workers = workers)

            try:
                return func(self, *args, **kwargs)    
            except Exception, e:
                logger.error(e)
                raise e
            finally:
                logger.info("Will close the thread pools")
                for pool in self.news_pools.values():
                    pool.shutdown(wait = True)
                    logger.info("pool closed successfully")
                
        return wrapped
    
    @__init_crawl    
    def crawl_sites(self, parse = True, download = True, nlp = True):

        self.do_parse = parse
        self.do_nlp   = nlp
        assert not (self.do_parse ^ self.do_nlp), """if nlp is set to true, parse must be set to true"""
        article_futures = []
        newspaper_config = self.crawler_config.crawl_option
        sources = {s.name : Source(s.url, newspaper_config) for s in self.crawler_config.sites}
        for s_name, source in sources.items():
            source.build()
            logger.info("Number of articles in newspaper %s is %d" %(s_name, source.size()))
            
        logger.info("Built the sources for the newspapers")
        if not download: return sources

        logger.info("downloading the article data from the newspapers")
        for s_name, source in sources.items():
            article_futures.extend([self.news_pools[s_name].submit(self.download_article, article)
                                    for article in source.articles])

        #download the actual content and parse 
        for future_obj in as_completed(article_futures):
            self.article_callback(future_obj)
        return sources
    
    
            
        
        
