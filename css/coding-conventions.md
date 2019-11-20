# CSS Conventions

## 1. Quy ước đặt tên

> Sử dụng format hyphens để đặt tên class.

```css
/* Correct */
.sec-nav

/* Wrong */
.sec_nav
.SecNav
.datasetlist {}
.datasetList {}
.dataset_list {}
```

- Luôn dùng danh từ để khai báo class/id
- Các danh từ phân tách bằng dấu ngạch ngang, mang ý nghĩa tương ứng level của các thẻ thao tác đến nó

```html
<div class="section-nav">
    <div class="section-nav-item">
    ...
    </div>
</div>
```

## 2. Giá trị

### 2.1 Font
> Luôn luôn khai báo font families có sẵn, ví dụ sans-serif hoặc serif.
```css
/* Correct */
font-family: "ff-din-web-1", Arial, Helvetica, sans-serif;

/* Wrong */
font-family: "ff-din-web-1";
```

### 2.3 Cú pháp CSS3
> Nếu sử dụng giá trị 0, thì không dùng đơn vị (px, em, etc.).

```css
/* Correct */
.nav a {
  padding: 5px 0 5px 2px;
}

/* Wrong */
.nav a {
  padding: 5px 0px 5px 2px;
}
```

### 2.3 Sử dụng mixin để  nhận về giá trị có tính logics, không sử dụng giá trị trong selector, những giá trị phải khai báo variables

```scss
/* =======================================================
   Wrong headers, block styles
   ======================================================= */
h1, h2, h3, h4, h5, h6 {
  margin-top: 0;
  margin-bottom: 0.5rem; 
}

p {
  margin-top: 0;
  margin-bottom: 1rem; 
}

/* =======================================================
   Correct headers, block styles
   ======================================================= */
/* _mixin.scss */
@mixin marin-block($top, $bottom) {
    margin-top: $top;
    margin-bottom: $bottom;
}

/* _variables.scss OR _common.scss */
$block-margin-top: 0;
$header-margin-bottom: 0.5rem;

/* _common.scss */
h1, h2, h3, h4, h5, h6 {
  @include marin-block($block-margin-top, $header-margin-bottom);
}

p {
  @include marin-block($block-margin-top, $header-margin-bottom*2);
}

```


## 3. Selectors

> Không sử dụng quá nhiều lớp cha và con
```css
/* Wrong */
.my-inbox .flyout-content .inner .message .inbox li div.take-action .actions ul li a {
}
```

### 3.1 Multiple selectors
> Khi sử dụng nhiều Selector, thì mỗi selector ở một dòng, cách nhau bằng dấu phẩy, và không có khoảng trắng ở cuối dấu phẩy

## 4. Comments

> Sử dụng comment cho `mỗi khối`, **phải** theo format như sau.

```css
/* ==========================================================================
   Section comment block
   ========================================================================== */
```

> Sử dụng comment để giải thích cho những code không rõ ràng, hoặc thêm comment nhiều nhất có thể.

```css
.prose p {
  font-size: 1.1666em /* 14px / 12px */;
}

.ie7 .search-form {
  /*
    Force the item to have layout in IE7 by setting display to block.
    See: http://reference.sitepoint.com/css/haslayout
  */
  display: inline-block;
}
```

5. Tổ chức file scss
> Phân chia các thư mục theo cấu trúc tương ứng với các chức năng
> Dùng file scss để  import các file scss ở thư mục con, tạo các file scss mới khi có sự thay đổi design, và import sử dụng file mới 

```bash
├── bootstrap
├── kakiage
│   ├── form
│   ├── page
│   │     ├── mypage
│   │     │     ├── _header.scss
│   │     │     ├── _header-version-1.scss
│   │     │     ├── _header-version-2.scss
│   │     │     └── _content.scss
│   │     ├── login
│   │     │     ├── _modal.scss
│   │     │     └── _input.scss
│   │     └── _page.scss
│   ├── mixins
│   ├── _pages.scss
│   └── _mixin.scss
└────styles.scss

```

```css
/* =======================================================
   styles.scss
   ======================================================= */
@import @"bootstrap/bootstrap";
@import @"kakiage/pages";


/* =======================================================
   kakiage/_pages.scss
   ======================================================= */
@import @"mixin";
@import @"variables";
@import @"page/page";

/* =======================================================
   kakiage/pages/_page.scss
   ======================================================= */
@import @"mypage/page/header";
@import @"mypage/page/content";
@import @"mypage/login/modal";
@import @"mypage/login/input";
```

