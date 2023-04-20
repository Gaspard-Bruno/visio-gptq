from enum import Enum
from dataclasses import dataclass
from typing import Optional, Any, List, Union, TypeVar, Type, Callable, cast
from datetime import datetime
import dateutil.parser


T = TypeVar("T")
EnumT = TypeVar("EnumT", bound=Enum)


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def to_enum(c: Type[EnumT], x: Any) -> EnumT:
    assert isinstance(x, c)
    return x.value


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


def is_type(t: Type[T], x: Any) -> T:
    assert isinstance(x, t)
    return x


def from_datetime(x: Any) -> datetime:
    return dateutil.parser.parse(x)


class AlternateLanguageLang(Enum):
    AR_AE = "ar-ae"
    PT_PT = "pt-pt"
    ZH_CN = "zh-cn"


class AlternateLanguageType(Enum):
    ARCHIVE = "archive"
    ARCHIVEPAGE = "archivepage"
    CAREERS = "careers"
    CASESTUDY = "casestudy"
    ETUDE = "etude"
    NAVIGATION = "navigation"
    PAGE = "page"


@dataclass
class AlternateLanguage:
    id: Optional[str] = None
    type: Optional[AlternateLanguageType] = None
    lang: Optional[AlternateLanguageLang] = None
    uid: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'AlternateLanguage':
        assert isinstance(obj, dict)
        id = from_union([from_str, from_none], obj.get("id"))
        type = from_union([AlternateLanguageType, from_none], obj.get("type"))
        lang = from_union([AlternateLanguageLang, from_none], obj.get("lang"))
        uid = from_union([from_str, from_none], obj.get("uid"))
        return AlternateLanguage(id, type, lang, uid)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.id is not None:
            result["id"] = from_union([from_str, from_none], self.id)
        if self.type is not None:
            result["type"] = from_union([lambda x: to_enum(AlternateLanguageType, x), from_none], self.type)
        if self.lang is not None:
            result["lang"] = from_union([lambda x: to_enum(AlternateLanguageLang, x), from_none], self.lang)
        if self.uid is not None:
            result["uid"] = from_union([from_str, from_none], self.uid)
        return result


class PolicyLinkLang(Enum):
    EN_GB = "en-gb"


class PolicyLinkLinkType(Enum):
    ANY = "Any"
    DOCUMENT = "Document"
    WEB = "Web"


class Target(Enum):
    BLANK = "_blank"


@dataclass
class PolicyLink:
    link_type: Optional[PolicyLinkLinkType] = None
    url: Optional[str] = None
    target: Optional[Target] = None
    id: Optional[str] = None
    type: Optional[AlternateLanguageType] = None
    tags: Optional[List[Any]] = None
    lang: Optional[PolicyLinkLang] = None
    slug: Optional[str] = None
    first_publication_date: Optional[str] = None
    last_publication_date: Optional[str] = None
    uid: Optional[str] = None
    is_broken: Optional[bool] = None

    @staticmethod
    def from_dict(obj: Any) -> 'PolicyLink':
        assert isinstance(obj, dict)
        link_type = from_union([PolicyLinkLinkType, from_none], obj.get("link_type"))
        url = from_union([from_str, from_none], obj.get("url"))
        target = from_union([Target, from_none], obj.get("target"))
        id = from_union([from_str, from_none], obj.get("id"))
        type = from_union([AlternateLanguageType, from_none], obj.get("type"))
        tags = from_union([lambda x: from_list(lambda x: x, x), from_none], obj.get("tags"))
        lang = from_union([PolicyLinkLang, from_none], obj.get("lang"))
        slug = from_union([from_str, from_none], obj.get("slug"))
        first_publication_date = from_union([from_str, from_none], obj.get("first_publication_date"))
        last_publication_date = from_union([from_str, from_none], obj.get("last_publication_date"))
        uid = from_union([from_str, from_none], obj.get("uid"))
        is_broken = from_union([from_bool, from_none], obj.get("isBroken"))
        return PolicyLink(link_type, url, target, id, type, tags, lang, slug, first_publication_date, last_publication_date, uid, is_broken)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.link_type is not None:
            result["link_type"] = from_union([lambda x: to_enum(PolicyLinkLinkType, x), from_none], self.link_type)
        if self.url is not None:
            result["url"] = from_union([from_str, from_none], self.url)
        if self.target is not None:
            result["target"] = from_union([lambda x: to_enum(Target, x), from_none], self.target)
        if self.id is not None:
            result["id"] = from_union([from_str, from_none], self.id)
        if self.type is not None:
            result["type"] = from_union([lambda x: to_enum(AlternateLanguageType, x), from_none], self.type)
        if self.tags is not None:
            result["tags"] = from_union([lambda x: from_list(lambda x: x, x), from_none], self.tags)
        if self.lang is not None:
            result["lang"] = from_union([lambda x: to_enum(PolicyLinkLang, x), from_none], self.lang)
        if self.slug is not None:
            result["slug"] = from_union([from_str, from_none], self.slug)
        if self.first_publication_date is not None:
            result["first_publication_date"] = from_union([from_str, from_none], self.first_publication_date)
        if self.last_publication_date is not None:
            result["last_publication_date"] = from_union([from_str, from_none], self.last_publication_date)
        if self.uid is not None:
            result["uid"] = from_union([from_str, from_none], self.uid)
        if self.is_broken is not None:
            result["isBroken"] = from_union([from_bool, from_none], self.is_broken)
        return result


class SpanType(Enum):
    EM = "em"
    HYPERLINK = "hyperlink"
    STRONG = "strong"


@dataclass
class Span:
    start: Optional[int] = None
    end: Optional[int] = None
    type: Optional[SpanType] = None
    data: Optional[PolicyLink] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Span':
        assert isinstance(obj, dict)
        start = from_union([from_int, from_none], obj.get("start"))
        end = from_union([from_int, from_none], obj.get("end"))
        type = from_union([SpanType, from_none], obj.get("type"))
        data = from_union([PolicyLink.from_dict, from_none], obj.get("data"))
        return Span(start, end, type, data)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.start is not None:
            result["start"] = from_union([from_int, from_none], self.start)
        if self.end is not None:
            result["end"] = from_union([from_int, from_none], self.end)
        if self.type is not None:
            result["type"] = from_union([lambda x: to_enum(SpanType, x), from_none], self.type)
        if self.data is not None:
            result["data"] = from_union([lambda x: to_class(PolicyLink, x), from_none], self.data)
        return result


class ButtonType(Enum):
    HEADING1 = "heading1"
    HEADING2 = "heading2"
    HEADING3 = "heading3"
    HEADING4 = "heading4"
    HEADING5 = "heading5"
    HEADING6 = "heading6"
    LIST_ITEM = "list-item"
    PARAGRAPH = "paragraph"


@dataclass
class Button:
    type: Optional[ButtonType] = None
    text: Optional[str] = None
    spans: Optional[List[Span]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Button':
        assert isinstance(obj, dict)
        type = from_union([ButtonType, from_none], obj.get("type"))
        text = from_union([from_str, from_none], obj.get("text"))
        spans = from_union([lambda x: from_list(Span.from_dict, x), from_none], obj.get("spans"))
        return Button(type, text, spans)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.type is not None:
            result["type"] = from_union([lambda x: to_enum(ButtonType, x), from_none], self.type)
        if self.text is not None:
            result["text"] = from_union([from_str, from_none], self.text)
        if self.spans is not None:
            result["spans"] = from_union([lambda x: from_list(lambda x: to_class(Span, x), x), from_none], self.spans)
        return result


@dataclass
class Address:
    address: Optional[List[Button]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Address':
        assert isinstance(obj, dict)
        address = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("address"))
        return Address(address)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.address is not None:
            result["address"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.address)
        return result


@dataclass
class Link:
    link_type: Optional[PolicyLinkLinkType] = None
    url: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Link':
        assert isinstance(obj, dict)
        link_type = from_union([PolicyLinkLinkType, from_none], obj.get("link_type"))
        url = from_union([from_str, from_none], obj.get("url"))
        return Link(link_type, url)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.link_type is not None:
            result["link_type"] = from_union([lambda x: to_enum(PolicyLinkLinkType, x), from_none], self.link_type)
        if self.url is not None:
            result["url"] = from_union([from_str, from_none], self.url)
        return result


class Kind(Enum):
    DOCUMENT = "document"
    IMAGE = "image"


class MediaLinkType(Enum):
    MEDIA = "Media"
    WEB = "Web"


@dataclass
class Media:
    size: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    link_type: Optional[MediaLinkType] = None
    name: Optional[str] = None
    kind: Optional[Kind] = None
    url: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Media':
        assert isinstance(obj, dict)
        size = from_union([from_none, lambda x: int(from_str(x))], obj.get("size"))
        height = from_union([from_none, lambda x: int(from_str(x))], obj.get("height"))
        width = from_union([from_none, lambda x: int(from_str(x))], obj.get("width"))
        link_type = from_union([MediaLinkType, from_none], obj.get("link_type"))
        name = from_union([from_str, from_none], obj.get("name"))
        kind = from_union([Kind, from_none], obj.get("kind"))
        url = from_union([from_str, from_none], obj.get("url"))
        return Media(size, height, width, link_type, name, kind, url)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.size is not None:
            result["size"] = from_union([lambda x: from_none((lambda x: is_type(type(None), x))(x)), lambda x: from_str((lambda x: str((lambda x: is_type(int, x))(x)))(x))], self.size)
        if self.height is not None:
            result["height"] = from_union([lambda x: from_none((lambda x: is_type(type(None), x))(x)), lambda x: from_str((lambda x: str((lambda x: is_type(int, x))(x)))(x))], self.height)
        if self.width is not None:
            result["width"] = from_union([lambda x: from_none((lambda x: is_type(type(None), x))(x)), lambda x: from_str((lambda x: str((lambda x: is_type(int, x))(x)))(x))], self.width)
        if self.link_type is not None:
            result["link_type"] = from_union([lambda x: to_enum(MediaLinkType, x), from_none], self.link_type)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.kind is not None:
            result["kind"] = from_union([lambda x: to_enum(Kind, x), from_none], self.kind)
        if self.url is not None:
            result["url"] = from_union([from_str, from_none], self.url)
        return result


@dataclass
class URL:
    link_type: Optional[PolicyLinkLinkType] = None
    url: Optional[str] = None
    target: Optional[Target] = None

    @staticmethod
    def from_dict(obj: Any) -> 'URL':
        assert isinstance(obj, dict)
        link_type = from_union([PolicyLinkLinkType, from_none], obj.get("link_type"))
        url = from_union([from_str, from_none], obj.get("url"))
        target = from_union([Target, from_none], obj.get("target"))
        return URL(link_type, url, target)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.link_type is not None:
            result["link_type"] = from_union([lambda x: to_enum(PolicyLinkLinkType, x), from_none], self.link_type)
        if self.url is not None:
            result["url"] = from_union([from_str, from_none], self.url)
        if self.target is not None:
            result["target"] = from_union([lambda x: to_enum(Target, x), from_none], self.target)
        return result


@dataclass
class Dimensions:
    width: Optional[int] = None
    height: Optional[int] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Dimensions':
        assert isinstance(obj, dict)
        width = from_union([from_int, from_none], obj.get("width"))
        height = from_union([from_int, from_none], obj.get("height"))
        return Dimensions(width, height)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.width is not None:
            result["width"] = from_union([from_int, from_none], self.width)
        if self.height is not None:
            result["height"] = from_union([from_int, from_none], self.height)
        return result


@dataclass
class SEOImage:
    alt: None
    copyright: None
    dimensions: Optional[Dimensions] = None
    url: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'SEOImage':
        assert isinstance(obj, dict)
        alt = from_none(obj.get("alt"))
        copyright = from_none(obj.get("copyright"))
        dimensions = from_union([Dimensions.from_dict, from_none], obj.get("dimensions"))
        url = from_union([from_str, from_none], obj.get("url"))
        return SEOImage(alt, copyright, dimensions, url)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.alt is not None:
            result["alt"] = from_none(self.alt)
        if self.copyright is not None:
            result["copyright"] = from_none(self.copyright)
        if self.dimensions is not None:
            result["dimensions"] = from_union([lambda x: to_class(Dimensions, x), from_none], self.dimensions)
        if self.url is not None:
            result["url"] = from_union([from_str, from_none], self.url)
        return result


class ImageSize(Enum):
    BIG = "Big"
    BIG_WITH_BACKGROUND = "Big with background"
    HALF = "Half"


class MediaType(Enum):
    IMAGE = "image"
    VIDEO = "video"


class Size(Enum):
    FULL = "Full"
    HALF = "Half"
    THE_6_COLUMM = "6 Columm"


@dataclass
class Item:
    title: Optional[List[Button]] = None
    categories: Optional[str] = None
    media: Optional[Media] = None
    media_type: Optional[MediaType] = None
    media_background: Optional[str] = None
    client: Optional[List[Button]] = None
    description: Optional[List[Button]] = None
    cta_title: Optional[List[Button]] = None
    cta_url: Optional[URL] = None
    page: Optional[PolicyLink] = None
    size: Optional[Size] = None
    button_text: Optional[List[Button]] = None
    introduction: Optional[List[Button]] = None
    links: Optional[List[Button]] = None
    tag_value: Optional[List[Button]] = None
    title1: Optional[List[Button]] = None
    list: Optional[List[Button]] = None
    content: Optional[List[Button]] = None
    job_title: Optional[List[Button]] = None
    job_description: Optional[List[Button]] = None
    number_of_open_positions: Optional[int] = None
    job_specs: Optional[List[Button]] = None
    apply_now_link: Optional[Link] = None
    image: Optional[SEOImage] = None
    image_size: Optional[ImageSize] = None
    background_color: Optional[str] = None
    client1: Optional[List[Button]] = None
    introduction1: Optional[List[Button]] = None
    cover: Optional[Media] = None
    button_url: Optional[PolicyLink] = None
    etude: Optional[PolicyLink] = None
    text_area: Optional[List[Button]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Item':
        assert isinstance(obj, dict)
        title = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("title"))
        categories = from_union([from_str, from_none], obj.get("categories"))
        media = from_union([Media.from_dict, from_none], obj.get("media"))
        media_type = from_union([MediaType, from_none], obj.get("media_type"))
        media_background = from_union([from_none, from_str], obj.get("media_background"))
        client = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("client"))
        description = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("description"))
        cta_title = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("cta_title"))
        cta_url = from_union([URL.from_dict, from_none], obj.get("cta_url"))
        page = from_union([PolicyLink.from_dict, from_none], obj.get("page"))
        size = from_union([Size, from_none], obj.get("size"))
        button_text = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("button_text"))
        introduction = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("introduction"))
        links = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("links"))
        tag_value = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("tag_value"))
        title1 = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("title1"))
        list = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("list"))
        content = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("content"))
        job_title = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("job_title"))
        job_description = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("job_description"))
        number_of_open_positions = from_union([from_int, from_none], obj.get("number_of_open_positions"))
        job_specs = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("job_specs"))
        apply_now_link = from_union([Link.from_dict, from_none], obj.get("apply_now_link"))
        image = from_union([SEOImage.from_dict, from_none], obj.get("image"))
        image_size = from_union([ImageSize, from_none], obj.get("image_size"))
        background_color = from_union([from_none, from_str], obj.get("background_color"))
        client1 = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("client1"))
        introduction1 = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("introduction1"))
        cover = from_union([Media.from_dict, from_none], obj.get("cover"))
        button_url = from_union([PolicyLink.from_dict, from_none], obj.get("button_url"))
        etude = from_union([PolicyLink.from_dict, from_none], obj.get("etude"))
        text_area = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("text_area"))
        return Item(title, categories, media, media_type, media_background, client, description, cta_title, cta_url, page, size, button_text, introduction, links, tag_value, title1, list, content, job_title, job_description, number_of_open_positions, job_specs, apply_now_link, image, image_size, background_color, client1, introduction1, cover, button_url, etude, text_area)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.title is not None:
            result["title"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.title)
        if self.categories is not None:
            result["categories"] = from_union([from_str, from_none], self.categories)
        if self.media is not None:
            result["media"] = from_union([lambda x: to_class(Media, x), from_none], self.media)
        if self.media_type is not None:
            result["media_type"] = from_union([lambda x: to_enum(MediaType, x), from_none], self.media_type)
        if self.media_background is not None:
            result["media_background"] = from_union([from_none, from_str], self.media_background)
        if self.client is not None:
            result["client"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.client)
        if self.description is not None:
            result["description"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.description)
        if self.cta_title is not None:
            result["cta_title"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.cta_title)
        if self.cta_url is not None:
            result["cta_url"] = from_union([lambda x: to_class(URL, x), from_none], self.cta_url)
        if self.page is not None:
            result["page"] = from_union([lambda x: to_class(PolicyLink, x), from_none], self.page)
        if self.size is not None:
            result["size"] = from_union([lambda x: to_enum(Size, x), from_none], self.size)
        if self.button_text is not None:
            result["button_text"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.button_text)
        if self.introduction is not None:
            result["introduction"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.introduction)
        if self.links is not None:
            result["links"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.links)
        if self.tag_value is not None:
            result["tag_value"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.tag_value)
        if self.title1 is not None:
            result["title1"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.title1)
        if self.list is not None:
            result["list"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.list)
        if self.content is not None:
            result["content"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.content)
        if self.job_title is not None:
            result["job_title"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.job_title)
        if self.job_description is not None:
            result["job_description"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.job_description)
        if self.number_of_open_positions is not None:
            result["number_of_open_positions"] = from_union([from_int, from_none], self.number_of_open_positions)
        if self.job_specs is not None:
            result["job_specs"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.job_specs)
        if self.apply_now_link is not None:
            result["apply_now_link"] = from_union([lambda x: to_class(Link, x), from_none], self.apply_now_link)
        if self.image is not None:
            result["image"] = from_union([lambda x: to_class(SEOImage, x), from_none], self.image)
        if self.image_size is not None:
            result["image_size"] = from_union([lambda x: to_enum(ImageSize, x), from_none], self.image_size)
        if self.background_color is not None:
            result["background_color"] = from_union([from_none, from_str], self.background_color)
        if self.client1 is not None:
            result["client1"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.client1)
        if self.introduction1 is not None:
            result["introduction1"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.introduction1)
        if self.cover is not None:
            result["cover"] = from_union([lambda x: to_class(Media, x), from_none], self.cover)
        if self.button_url is not None:
            result["button_url"] = from_union([lambda x: to_class(PolicyLink, x), from_none], self.button_url)
        if self.etude is not None:
            result["etude"] = from_union([lambda x: to_class(PolicyLink, x), from_none], self.etude)
        if self.text_area is not None:
            result["text_area"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.text_area)
        return result


@dataclass
class LottieFile:
    size: Optional[int] = None
    link_type: Optional[MediaLinkType] = None
    name: Optional[str] = None
    kind: Optional[Kind] = None
    url: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'LottieFile':
        assert isinstance(obj, dict)
        size = from_union([from_none, lambda x: int(from_str(x))], obj.get("size"))
        link_type = from_union([MediaLinkType, from_none], obj.get("link_type"))
        name = from_union([from_str, from_none], obj.get("name"))
        kind = from_union([Kind, from_none], obj.get("kind"))
        url = from_union([from_str, from_none], obj.get("url"))
        return LottieFile(size, link_type, name, kind, url)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.size is not None:
            result["size"] = from_union([lambda x: from_none((lambda x: is_type(type(None), x))(x)), lambda x: from_str((lambda x: str((lambda x: is_type(int, x))(x)))(x))], self.size)
        if self.link_type is not None:
            result["link_type"] = from_union([lambda x: to_enum(MediaLinkType, x), from_none], self.link_type)
        if self.name is not None:
            result["name"] = from_union([from_str, from_none], self.name)
        if self.kind is not None:
            result["kind"] = from_union([lambda x: to_enum(Kind, x), from_none], self.kind)
        if self.url is not None:
            result["url"] = from_union([from_str, from_none], self.url)
        return result


@dataclass
class Primary:
    text_color: None
    background_color: None
    sub_title: Optional[List[Button]] = None
    heading: Optional[List[Button]] = None
    type: Optional[str] = None
    intro_text: Optional[List[Button]] = None
    media1: Optional[Media] = None
    media_type: Optional[str] = None
    text_color1: Optional[str] = None
    background_color1: Optional[str] = None
    big_text: Optional[List[Button]] = None
    textaarea: Optional[List[Button]] = None
    media: Optional[Media] = None
    media_size: Optional[str] = None
    backgroundcolor: Optional[str] = None
    textcolor: Optional[str] = None
    style: Optional[str] = None
    cta_title: Optional[List[Any]] = None
    cta_url: Optional[PolicyLink] = None
    tag_list_type: Optional[List[Button]] = None
    link_name: Optional[List[Button]] = None
    link_url: Optional[PolicyLink] = None
    title1: Optional[List[Button]] = None
    scope: Optional[List[Button]] = None
    show_jobs_counter: Optional[bool] = None
    text_area: Optional[List[Button]] = None
    size: Optional[str] = None
    vertical_aligment: Optional[str] = None
    horizontal_aligment: Optional[str] = None
    title: Optional[List[Button]] = None
    description: Optional[List[Any]] = None
    lottie_file: Optional[LottieFile] = None
    video_url: Optional[LottieFile] = None
    video_source: Optional[str] = None
    video_poster: Optional[Media] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Primary':
        assert isinstance(obj, dict)
        text_color = from_none(obj.get("text_color"))
        background_color = from_none(obj.get("background_color"))
        sub_title = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("sub_title"))
        heading = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("heading"))
        type = from_union([from_str, from_none], obj.get("type"))
        intro_text = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("intro_text"))
        media1 = from_union([Media.from_dict, from_none], obj.get("media1"))
        media_type = from_union([from_str, from_none], obj.get("media_type"))
        text_color1 = from_union([from_none, from_str], obj.get("text_color1"))
        background_color1 = from_union([from_none, from_str], obj.get("background_color1"))
        big_text = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("big_text"))
        textaarea = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("textaarea"))
        media = from_union([Media.from_dict, from_none], obj.get("media"))
        media_size = from_union([from_none, from_str], obj.get("media_size"))
        backgroundcolor = from_union([from_none, from_str], obj.get("backgroundcolor"))
        textcolor = from_union([from_none, from_str], obj.get("textcolor"))
        style = from_union([from_str, from_none], obj.get("style"))
        cta_title = from_union([lambda x: from_list(lambda x: x, x), from_none], obj.get("cta_title"))
        cta_url = from_union([PolicyLink.from_dict, from_none], obj.get("cta_url"))
        tag_list_type = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("tag_list_type"))
        link_name = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("link_name"))
        link_url = from_union([PolicyLink.from_dict, from_none], obj.get("link_url"))
        title1 = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("title1"))
        scope = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("scope"))
        show_jobs_counter = from_union([from_bool, from_none], obj.get("show_jobs_counter"))
        text_area = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("text_area"))
        size = from_union([from_str, from_none], obj.get("size"))
        vertical_aligment = from_union([from_str, from_none], obj.get("vertical_aligment"))
        horizontal_aligment = from_union([from_str, from_none], obj.get("horizontal_aligment"))
        title = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("title"))
        description = from_union([lambda x: from_list(lambda x: x, x), from_none], obj.get("description"))
        lottie_file = from_union([LottieFile.from_dict, from_none], obj.get("lottie_file"))
        video_url = from_union([LottieFile.from_dict, from_none], obj.get("video_url"))
        video_source = from_union([from_str, from_none], obj.get("video_source"))
        video_poster = from_union([Media.from_dict, from_none], obj.get("video_poster"))
        return Primary(text_color, background_color, sub_title, heading, type, intro_text, media1, media_type, text_color1, background_color1, big_text, textaarea, media, media_size, backgroundcolor, textcolor, style, cta_title, cta_url, tag_list_type, link_name, link_url, title1, scope, show_jobs_counter, text_area, size, vertical_aligment, horizontal_aligment, title, description, lottie_file, video_url, video_source, video_poster)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.text_color is not None:
            result["text_color"] = from_none(self.text_color)
        if self.background_color is not None:
            result["background_color"] = from_none(self.background_color)
        if self.sub_title is not None:
            result["sub_title"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.sub_title)
        if self.heading is not None:
            result["heading"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.heading)
        if self.type is not None:
            result["type"] = from_union([from_str, from_none], self.type)
        if self.intro_text is not None:
            result["intro_text"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.intro_text)
        if self.media1 is not None:
            result["media1"] = from_union([lambda x: to_class(Media, x), from_none], self.media1)
        if self.media_type is not None:
            result["media_type"] = from_union([from_str, from_none], self.media_type)
        if self.text_color1 is not None:
            result["text_color1"] = from_union([from_none, from_str], self.text_color1)
        if self.background_color1 is not None:
            result["background_color1"] = from_union([from_none, from_str], self.background_color1)
        if self.big_text is not None:
            result["big_text"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.big_text)
        if self.textaarea is not None:
            result["textaarea"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.textaarea)
        if self.media is not None:
            result["media"] = from_union([lambda x: to_class(Media, x), from_none], self.media)
        if self.media_size is not None:
            result["media_size"] = from_union([from_none, from_str], self.media_size)
        if self.backgroundcolor is not None:
            result["backgroundcolor"] = from_union([from_none, from_str], self.backgroundcolor)
        if self.textcolor is not None:
            result["textcolor"] = from_union([from_none, from_str], self.textcolor)
        if self.style is not None:
            result["style"] = from_union([from_str, from_none], self.style)
        if self.cta_title is not None:
            result["cta_title"] = from_union([lambda x: from_list(lambda x: x, x), from_none], self.cta_title)
        if self.cta_url is not None:
            result["cta_url"] = from_union([lambda x: to_class(PolicyLink, x), from_none], self.cta_url)
        if self.tag_list_type is not None:
            result["tag_list_type"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.tag_list_type)
        if self.link_name is not None:
            result["link_name"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.link_name)
        if self.link_url is not None:
            result["link_url"] = from_union([lambda x: to_class(PolicyLink, x), from_none], self.link_url)
        if self.title1 is not None:
            result["title1"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.title1)
        if self.scope is not None:
            result["scope"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.scope)
        if self.show_jobs_counter is not None:
            result["show_jobs_counter"] = from_union([from_bool, from_none], self.show_jobs_counter)
        if self.text_area is not None:
            result["text_area"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.text_area)
        if self.size is not None:
            result["size"] = from_union([from_str, from_none], self.size)
        if self.vertical_aligment is not None:
            result["vertical_aligment"] = from_union([from_str, from_none], self.vertical_aligment)
        if self.horizontal_aligment is not None:
            result["horizontal_aligment"] = from_union([from_str, from_none], self.horizontal_aligment)
        if self.title is not None:
            result["title"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.title)
        if self.description is not None:
            result["description"] = from_union([lambda x: from_list(lambda x: x, x), from_none], self.description)
        if self.lottie_file is not None:
            result["lottie_file"] = from_union([lambda x: to_class(LottieFile, x), from_none], self.lottie_file)
        if self.video_url is not None:
            result["video_url"] = from_union([lambda x: to_class(LottieFile, x), from_none], self.video_url)
        if self.video_source is not None:
            result["video_source"] = from_union([from_str, from_none], self.video_source)
        if self.video_poster is not None:
            result["video_poster"] = from_union([lambda x: to_class(Media, x), from_none], self.video_poster)
        return result


@dataclass
class Body:
    slice_label: None
    primary: Optional[Primary] = None
    items: Optional[List[Item]] = None
    id: Optional[str] = None
    slice_type: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Body':
        assert isinstance(obj, dict)
        slice_label = from_none(obj.get("slice_label"))
        primary = from_union([Primary.from_dict, from_none], obj.get("primary"))
        items = from_union([lambda x: from_list(Item.from_dict, x), from_none], obj.get("items"))
        id = from_union([from_str, from_none], obj.get("id"))
        slice_type = from_union([from_str, from_none], obj.get("slice_type"))
        return Body(slice_label, primary, items, id, slice_type)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.slice_label is not None:
            result["slice_label"] = from_none(self.slice_label)
        if self.primary is not None:
            result["primary"] = from_union([lambda x: to_class(Primary, x), from_none], self.primary)
        if self.items is not None:
            result["items"] = from_union([lambda x: from_list(lambda x: to_class(Item, x), x), from_none], self.items)
        if self.id is not None:
            result["id"] = from_union([from_str, from_none], self.id)
        if self.slice_type is not None:
            result["slice_type"] = from_union([from_str, from_none], self.slice_type)
        return result


@dataclass
class Cover:
    link_type: Optional[PolicyLinkLinkType] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Cover':
        assert isinstance(obj, dict)
        link_type = from_union([PolicyLinkLinkType, from_none], obj.get("link_type"))
        return Cover(link_type)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.link_type is not None:
            result["link_type"] = from_union([lambda x: to_enum(PolicyLinkLinkType, x), from_none], self.link_type)
        return result


@dataclass
class Credit:
    title1: Optional[List[Button]] = None
    text: Optional[List[Button]] = None
    title: Optional[List[Button]] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Credit':
        assert isinstance(obj, dict)
        title1 = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("title1"))
        text = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("text"))
        title = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("title"))
        return Credit(title1, text, title)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.title1 is not None:
            result["title1"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.title1)
        if self.text is not None:
            result["text"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.text)
        if self.title is not None:
            result["title"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.title)
        return result


@dataclass
class ExternalLink:
    text: Optional[List[Button]] = None
    link: Optional[Link] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ExternalLink':
        assert isinstance(obj, dict)
        text = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("text"))
        link = from_union([Link.from_dict, from_none], obj.get("link"))
        return ExternalLink(text, link)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.text is not None:
            result["text"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.text)
        if self.link is not None:
            result["link"] = from_union([lambda x: to_class(Link, x), from_none], self.link)
        return result


@dataclass
class FilterLabelsOverwrite:
    original: Optional[str] = None
    overwrite_text: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'FilterLabelsOverwrite':
        assert isinstance(obj, dict)
        original = from_union([from_str, from_none], obj.get("original"))
        overwrite_text = from_union([from_str, from_none], obj.get("overwrite_text"))
        return FilterLabelsOverwrite(original, overwrite_text)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.original is not None:
            result["original"] = from_union([from_str, from_none], self.original)
        if self.overwrite_text is not None:
            result["overwrite_text"] = from_union([from_str, from_none], self.overwrite_text)
        return result


@dataclass
class FooterLink:
    link_name: Optional[List[Button]] = None
    link_url: Optional[PolicyLink] = None

    @staticmethod
    def from_dict(obj: Any) -> 'FooterLink':
        assert isinstance(obj, dict)
        link_name = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("link_name"))
        link_url = from_union([PolicyLink.from_dict, from_none], obj.get("link_url"))
        return FooterLink(link_name, link_url)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.link_name is not None:
            result["link_name"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.link_name)
        if self.link_url is not None:
            result["link_url"] = from_union([lambda x: to_class(PolicyLink, x), from_none], self.link_url)
        return result


@dataclass
class FooterSocialMediaLink:
    link_name: Optional[List[Button]] = None
    link_url: Optional[URL] = None

    @staticmethod
    def from_dict(obj: Any) -> 'FooterSocialMediaLink':
        assert isinstance(obj, dict)
        link_name = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("link_name"))
        link_url = from_union([URL.from_dict, from_none], obj.get("link_url"))
        return FooterSocialMediaLink(link_name, link_url)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.link_name is not None:
            result["link_name"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.link_name)
        if self.link_url is not None:
            result["link_url"] = from_union([lambda x: to_class(URL, x), from_none], self.link_url)
        return result


@dataclass
class ProjectColor:
    color: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'ProjectColor':
        assert isinstance(obj, dict)
        color = from_union([from_str, from_none], obj.get("color"))
        return ProjectColor(color)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.color is not None:
            result["color"] = from_union([from_str, from_none], self.color)
        return result


@dataclass
class Data:
    footer_background_color: None
    client: Optional[Union[List[Button], str]] = None
    text_color: Optional[str] = None
    background_color: Optional[str] = None
    body: Optional[List[Body]] = None
    seo_title: Optional[List[Button]] = None
    seo_meta: Optional[List[Button]] = None
    seo_image: Optional[SEOImage] = None
    client_name: Optional[List[Button]] = None
    title: Optional[List[Button]] = None
    introduction: Optional[List[Button]] = None
    active: Optional[bool] = None
    thumbnail: Optional[SEOImage] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    cover: Optional[Cover] = None
    credits: Optional[List[Credit]] = None
    external_links: Optional[List[ExternalLink]] = None
    project_colors: Optional[List[ProjectColor]] = None
    email: Optional[List[Button]] = None
    footer_text: Optional[List[Button]] = None
    footer_links: Optional[List[FooterLink]] = None
    footer_social_media_links: Optional[List[FooterSocialMediaLink]] = None
    addresses: Optional[List[Address]] = None
    message: Optional[List[Button]] = None
    policy_text: Optional[List[Button]] = None
    policy_link: Optional[PolicyLink] = None
    button: Optional[List[Button]] = None
    show_date_range: Optional[bool] = None
    show_filter: Optional[bool] = None
    filter_labels_overwrite: Optional[List[FilterLabelsOverwrite]] = None
    year: Optional[str] = None
    project: Optional[str] = None
    release_date: Optional[datetime] = None
    media: Optional[Media] = None
    thumbnail_size: Optional[str] = None
    thumbnail_aligment: Optional[str] = None
    hide_from_home: Optional[bool] = None
    media1: Optional[Media] = None
    categories: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Data':
        assert isinstance(obj, dict)
        footer_background_color = from_none(obj.get("footer_background_color"))
        client = from_union([lambda x: from_list(Button.from_dict, x), from_str, from_none], obj.get("client"))
        text_color = from_union([from_none, from_str], obj.get("text_color"))
        background_color = from_union([from_none, from_str], obj.get("background_color"))
        body = from_union([lambda x: from_list(Body.from_dict, x), from_none], obj.get("body"))
        seo_title = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("seo_title"))
        seo_meta = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("seo_meta"))
        seo_image = from_union([SEOImage.from_dict, from_none], obj.get("seo_image"))
        client_name = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("client_name"))
        title = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("title"))
        introduction = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("introduction"))
        active = from_union([from_bool, from_none], obj.get("active"))
        thumbnail = from_union([SEOImage.from_dict, from_none], obj.get("thumbnail"))
        start_date = from_union([from_datetime, from_none], obj.get("start_date"))
        end_date = from_union([from_datetime, from_none], obj.get("end_date"))
        cover = from_union([Cover.from_dict, from_none], obj.get("cover"))
        credits = from_union([lambda x: from_list(Credit.from_dict, x), from_none], obj.get("credits"))
        external_links = from_union([lambda x: from_list(ExternalLink.from_dict, x), from_none], obj.get("external_links"))
        project_colors = from_union([lambda x: from_list(ProjectColor.from_dict, x), from_none], obj.get("project_colors"))
        email = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("email"))
        footer_text = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("footer_text"))
        footer_links = from_union([lambda x: from_list(FooterLink.from_dict, x), from_none], obj.get("footer_links"))
        footer_social_media_links = from_union([lambda x: from_list(FooterSocialMediaLink.from_dict, x), from_none], obj.get("footer_social_media_links"))
        addresses = from_union([lambda x: from_list(Address.from_dict, x), from_none], obj.get("addresses"))
        message = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("message"))
        policy_text = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("policy_text"))
        policy_link = from_union([PolicyLink.from_dict, from_none], obj.get("policy_link"))
        button = from_union([lambda x: from_list(Button.from_dict, x), from_none], obj.get("button"))
        show_date_range = from_union([from_bool, from_none], obj.get("show_date_range"))
        show_filter = from_union([from_bool, from_none], obj.get("show_filter"))
        filter_labels_overwrite = from_union([lambda x: from_list(FilterLabelsOverwrite.from_dict, x), from_none], obj.get("filter_labels_overwrite"))
        year = from_union([from_str, from_none], obj.get("year"))
        project = from_union([from_str, from_none], obj.get("project"))
        release_date = from_union([from_datetime, from_none], obj.get("release_date"))
        media = from_union([Media.from_dict, from_none], obj.get("media"))
        thumbnail_size = from_union([from_str, from_none], obj.get("thumbnail_size"))
        thumbnail_aligment = from_union([from_none, from_str], obj.get("thumbnail_aligment"))
        hide_from_home = from_union([from_bool, from_none], obj.get("hide_from_home"))
        media1 = from_union([Media.from_dict, from_none], obj.get("media1"))
        categories = from_union([from_str, from_none], obj.get("categories"))
        return Data(footer_background_color, client, text_color, background_color, body, seo_title, seo_meta, seo_image, client_name, title, introduction, active, thumbnail, start_date, end_date, cover, credits, external_links, project_colors, email, footer_text, footer_links, footer_social_media_links, addresses, message, policy_text, policy_link, button, show_date_range, show_filter, filter_labels_overwrite, year, project, release_date, media, thumbnail_size, thumbnail_aligment, hide_from_home, media1, categories)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.footer_background_color is not None:
            result["footer_background_color"] = from_none(self.footer_background_color)
        if self.client is not None:
            result["client"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_str, from_none], self.client)
        if self.text_color is not None:
            result["text_color"] = from_union([from_none, from_str], self.text_color)
        if self.background_color is not None:
            result["background_color"] = from_union([from_none, from_str], self.background_color)
        if self.body is not None:
            result["body"] = from_union([lambda x: from_list(lambda x: to_class(Body, x), x), from_none], self.body)
        if self.seo_title is not None:
            result["seo_title"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.seo_title)
        if self.seo_meta is not None:
            result["seo_meta"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.seo_meta)
        if self.seo_image is not None:
            result["seo_image"] = from_union([lambda x: to_class(SEOImage, x), from_none], self.seo_image)
        if self.client_name is not None:
            result["client_name"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.client_name)
        if self.title is not None:
            result["title"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.title)
        if self.introduction is not None:
            result["introduction"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.introduction)
        if self.active is not None:
            result["active"] = from_union([from_bool, from_none], self.active)
        if self.thumbnail is not None:
            result["thumbnail"] = from_union([lambda x: to_class(SEOImage, x), from_none], self.thumbnail)
        if self.start_date is not None:
            result["start_date"] = from_union([lambda x: x.isoformat(), from_none], self.start_date)
        if self.end_date is not None:
            result["end_date"] = from_union([lambda x: x.isoformat(), from_none], self.end_date)
        if self.cover is not None:
            result["cover"] = from_union([lambda x: to_class(Cover, x), from_none], self.cover)
        if self.credits is not None:
            result["credits"] = from_union([lambda x: from_list(lambda x: to_class(Credit, x), x), from_none], self.credits)
        if self.external_links is not None:
            result["external_links"] = from_union([lambda x: from_list(lambda x: to_class(ExternalLink, x), x), from_none], self.external_links)
        if self.project_colors is not None:
            result["project_colors"] = from_union([lambda x: from_list(lambda x: to_class(ProjectColor, x), x), from_none], self.project_colors)
        if self.email is not None:
            result["email"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.email)
        if self.footer_text is not None:
            result["footer_text"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.footer_text)
        if self.footer_links is not None:
            result["footer_links"] = from_union([lambda x: from_list(lambda x: to_class(FooterLink, x), x), from_none], self.footer_links)
        if self.footer_social_media_links is not None:
            result["footer_social_media_links"] = from_union([lambda x: from_list(lambda x: to_class(FooterSocialMediaLink, x), x), from_none], self.footer_social_media_links)
        if self.addresses is not None:
            result["addresses"] = from_union([lambda x: from_list(lambda x: to_class(Address, x), x), from_none], self.addresses)
        if self.message is not None:
            result["message"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.message)
        if self.policy_text is not None:
            result["policy_text"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.policy_text)
        if self.policy_link is not None:
            result["policy_link"] = from_union([lambda x: to_class(PolicyLink, x), from_none], self.policy_link)
        if self.button is not None:
            result["button"] = from_union([lambda x: from_list(lambda x: to_class(Button, x), x), from_none], self.button)
        if self.show_date_range is not None:
            result["show_date_range"] = from_union([from_bool, from_none], self.show_date_range)
        if self.show_filter is not None:
            result["show_filter"] = from_union([from_bool, from_none], self.show_filter)
        if self.filter_labels_overwrite is not None:
            result["filter_labels_overwrite"] = from_union([lambda x: from_list(lambda x: to_class(FilterLabelsOverwrite, x), x), from_none], self.filter_labels_overwrite)
        if self.year is not None:
            result["year"] = from_union([from_str, from_none], self.year)
        if self.project is not None:
            result["project"] = from_union([from_str, from_none], self.project)
        if self.release_date is not None:
            result["release_date"] = from_union([lambda x: x.isoformat(), from_none], self.release_date)
        if self.media is not None:
            result["media"] = from_union([lambda x: to_class(Media, x), from_none], self.media)
        if self.thumbnail_size is not None:
            result["thumbnail_size"] = from_union([from_str, from_none], self.thumbnail_size)
        if self.thumbnail_aligment is not None:
            result["thumbnail_aligment"] = from_union([from_none, from_str], self.thumbnail_aligment)
        if self.hide_from_home is not None:
            result["hide_from_home"] = from_union([from_bool, from_none], self.hide_from_home)
        if self.media1 is not None:
            result["media1"] = from_union([lambda x: to_class(Media, x), from_none], self.media1)
        if self.categories is not None:
            result["categories"] = from_union([from_str, from_none], self.categories)
        return result


@dataclass
class Result:
    url: None
    id: Optional[str] = None
    uid: Optional[str] = None
    type: Optional[AlternateLanguageType] = None
    href: Optional[str] = None
    tags: Optional[List[Any]] = None
    first_publication_date: Optional[str] = None
    last_publication_date: Optional[str] = None
    slugs: Optional[List[str]] = None
    linked_documents: Optional[List[Any]] = None
    lang: Optional[PolicyLinkLang] = None
    alternate_languages: Optional[List[AlternateLanguage]] = None
    data: Optional[Data] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Result':
        assert isinstance(obj, dict)
        url = from_none(obj.get("url"))
        id = from_union([from_str, from_none], obj.get("id"))
        uid = from_union([from_str, from_none], obj.get("uid"))
        type = from_union([AlternateLanguageType, from_none], obj.get("type"))
        href = from_union([from_str, from_none], obj.get("href"))
        tags = from_union([lambda x: from_list(lambda x: x, x), from_none], obj.get("tags"))
        first_publication_date = from_union([from_str, from_none], obj.get("first_publication_date"))
        last_publication_date = from_union([from_str, from_none], obj.get("last_publication_date"))
        slugs = from_union([lambda x: from_list(from_str, x), from_none], obj.get("slugs"))
        linked_documents = from_union([lambda x: from_list(lambda x: x, x), from_none], obj.get("linked_documents"))
        lang = from_union([PolicyLinkLang, from_none], obj.get("lang"))
        alternate_languages = from_union([lambda x: from_list(AlternateLanguage.from_dict, x), from_none], obj.get("alternate_languages"))
        data = from_union([Data.from_dict, from_none], obj.get("data"))
        return Result(url, id, uid, type, href, tags, first_publication_date, last_publication_date, slugs, linked_documents, lang, alternate_languages, data)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.url is not None:
            result["url"] = from_none(self.url)
        if self.id is not None:
            result["id"] = from_union([from_str, from_none], self.id)
        if self.uid is not None:
            result["uid"] = from_union([from_str, from_none], self.uid)
        if self.type is not None:
            result["type"] = from_union([lambda x: to_enum(AlternateLanguageType, x), from_none], self.type)
        if self.href is not None:
            result["href"] = from_union([from_str, from_none], self.href)
        if self.tags is not None:
            result["tags"] = from_union([lambda x: from_list(lambda x: x, x), from_none], self.tags)
        if self.first_publication_date is not None:
            result["first_publication_date"] = from_union([from_str, from_none], self.first_publication_date)
        if self.last_publication_date is not None:
            result["last_publication_date"] = from_union([from_str, from_none], self.last_publication_date)
        if self.slugs is not None:
            result["slugs"] = from_union([lambda x: from_list(from_str, x), from_none], self.slugs)
        if self.linked_documents is not None:
            result["linked_documents"] = from_union([lambda x: from_list(lambda x: x, x), from_none], self.linked_documents)
        if self.lang is not None:
            result["lang"] = from_union([lambda x: to_enum(PolicyLinkLang, x), from_none], self.lang)
        if self.alternate_languages is not None:
            result["alternate_languages"] = from_union([lambda x: from_list(lambda x: to_class(AlternateLanguage, x), x), from_none], self.alternate_languages)
        if self.data is not None:
            result["data"] = from_union([lambda x: to_class(Data, x), from_none], self.data)
        return result


@dataclass
class Dataset:
    prev_page: None
    page: Optional[int] = None
    results_per_page: Optional[int] = None
    results_size: Optional[int] = None
    total_results_size: Optional[int] = None
    total_pages: Optional[int] = None
    next_page: Optional[str] = None
    results: Optional[List[Result]] = None
    version: Optional[str] = None
    license: Optional[str] = None

    @staticmethod
    def from_dict(obj: Any) -> 'Dataset':
        assert isinstance(obj, dict)
        prev_page = from_none(obj.get("prev_page"))
        page = from_union([from_int, from_none], obj.get("page"))
        results_per_page = from_union([from_int, from_none], obj.get("results_per_page"))
        results_size = from_union([from_int, from_none], obj.get("results_size"))
        total_results_size = from_union([from_int, from_none], obj.get("total_results_size"))
        total_pages = from_union([from_int, from_none], obj.get("total_pages"))
        next_page = from_union([from_str, from_none], obj.get("next_page"))
        results = from_union([lambda x: from_list(Result.from_dict, x), from_none], obj.get("results"))
        version = from_union([from_str, from_none], obj.get("version"))
        license = from_union([from_str, from_none], obj.get("license"))
        return Dataset(prev_page, page, results_per_page, results_size, total_results_size, total_pages, next_page, results, version, license)

    def to_dict(self) -> dict:
        result: dict = {}
        if self.prev_page is not None:
            result["prev_page"] = from_none(self.prev_page)
        if self.page is not None:
            result["page"] = from_union([from_int, from_none], self.page)
        if self.results_per_page is not None:
            result["results_per_page"] = from_union([from_int, from_none], self.results_per_page)
        if self.results_size is not None:
            result["results_size"] = from_union([from_int, from_none], self.results_size)
        if self.total_results_size is not None:
            result["total_results_size"] = from_union([from_int, from_none], self.total_results_size)
        if self.total_pages is not None:
            result["total_pages"] = from_union([from_int, from_none], self.total_pages)
        if self.next_page is not None:
            result["next_page"] = from_union([from_str, from_none], self.next_page)
        if self.results is not None:
            result["results"] = from_union([lambda x: from_list(lambda x: to_class(Result, x), x), from_none], self.results)
        if self.version is not None:
            result["version"] = from_union([from_str, from_none], self.version)
        if self.license is not None:
            result["license"] = from_union([from_str, from_none], self.license)
        return result


def dataset_from_dict(s: Any) -> Dataset:
    return Dataset.from_dict(s)


def dataset_to_dict(x: Dataset) -> Any:
    return to_class(Dataset, x)
