from pytube import YouTube


class YTDownloader:
    """
    Youtube downloader
    Args:
        path: path to directory where video will be saved
    """
    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def url(yt_id: str) -> str:
        """
        Get full url from Youtube id
        Args:
            yt_id: Youtube id of the video

        Returns:
            full path to video
        """
        return "https://www.youtube.com/watch?v={}".format(yt_id)

    def download(self, yt_id: str, name: str = "YT_Video"):
        """
        Download the video
        Args:
            yt_id: Youtube id of the video
            name: additional name in file name (format: <name> <yt_id>.mp4)
        """
        url = self.url(yt_id)

        try:
            video = YouTube(url)
        except:
            print("Video not found")
            return

        video = video.streams.first()
        video.download(self.path, filename="{} {}".format(name, yt_id))
