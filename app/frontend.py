import os
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener


# iPhone photos AirDropped to the Mac arrive as .heic, which Pillow (and so
# st.image) cannot open on its own. One call at import fixes that for the
# whole process. Registering twice is harmless.
register_heif_opener()


API_URL = os.getenv("API_URL")

# Must be the FIRST Streamlit command executed -- page title (browser tab)
# and favicon, the icon cut from the logo's brick mark.
st.set_page_config(
    page_title="LegoFitter",
    page_icon=str(Path(__file__).parent / "assets" / "favicon.png"),
    layout="centered",
)


@st.cache_data(show_spinner=False)
def load_part_image(url: str) -> bytes | None:
    """Download a Rebrickable part photo and make its white background
    transparent, so the brick floats on the page instead of sitting in a
    stark white tile.

    Flood fill from the four corners, not a global threshold: only white
    that is CONNECTED to the border becomes transparent, so a white brick
    keeps its own white pixels. thresh=40 also catches the off-white
    shadow gradient around Rebrickable's studio shots.

    Cached by URL (Streamlit reruns this whole script on every click);
    returns None on any failure so a broken image never breaks the page.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        picture = Image.open(BytesIO(response.content)).convert("RGBA")

        corners = [
            (0, 0),
            (picture.width - 1, 0),
            (0, picture.height - 1),
            (picture.width - 1, picture.height - 1),
        ]
        for corner in corners:
            ImageDraw.floodfill(picture, corner, (0, 0, 0, 0), thresh=40)

        output = BytesIO()
        picture.save(output, format="PNG")
        return output.getvalue()

    except Exception:
        return None


# Big square validation checkboxes -- Streamlit's are tiny by default.
# If a Streamlit update renames these test-ids, the CSS silently stops
# applying and the checkboxes still work at their normal size.
st.markdown(
    """
    <style>
    div[data-testid="stCheckbox"] label > span:first-of-type {
        transform: scale(1.6);
        transform-origin: left center;
        margin-right: 0.6rem;
    }
    div[data-testid="stCheckbox"] p {
        font-size: 1.05rem;
        font-weight: 600;
    }

    /* Results pane: photo + checklist side by side. The page stays a
       narrow centred column everywhere else; only the block holding the
       .lf-breakout marker spans the full window (photo half, list
       half), so the photo is big enough to read the labels next to
       the list. */
    div[data-testid="stHorizontalBlock"]:has(.lf-breakout) {
        /* !important: Streamlit writes its measured width as an inline
           style on every block, which would otherwise win and leave
           the block half as wide as intended. */
        width: calc(100vw - 4rem) !important;
        max-width: none !important;
        margin-left: calc(50% - 50vw + 2rem);
    }
    div[data-testid="stHorizontalBlock"]:has(.lf-breakout)
        > div[data-testid="stColumn"] {
        flex: 1 1 0 !important;
        width: auto !important;
        min-width: 0;
    }
    /* Checklist packed tight so ten bricks plus the button fit in one
       screen next to the photo: small row gap, labels never wrap. */
    div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {
        gap: var(--lf-row-gap);
    }
    div[data-testid="stCheckbox"] label {
        white-space: nowrap;
        overflow: visible;
    }
    /* The photo column stays put while the checklist scrolls past it.
       align-self:flex-start is what makes sticky work: a stretched
       column is as tall as the list and has nowhere to stick. */
    div[data-testid="stColumn"]:has(.lf-breakout) {
        position: sticky;
        top: 3.75rem;
        align-self: flex-start;
    }
    /* One shared height for the pane: the photo is capped to it (a
       portrait phone shot would otherwise run off the screen) and
       left-aligned under its title. The catalog pictures in the
       checklist are sized so that N rows plus their gaps add up to
       the same height minus the "Tout correct" line -- the last brick
       ends level with the photo's bottom edge. --lf-rows (N) is set
       per analysis next to the list. */
    :root { --lf-pane-h: calc(100vh - 11rem); --lf-rows: 10; --lf-row-gap: 0.35rem; }
    div[data-testid="stColumn"]:has(.lf-breakout) div[data-testid="stImage"] {
        justify-content: flex-start;
        align-items: flex-start;
    }
    div[data-testid="stColumn"]:has(.lf-breakout) div[data-testid="stImage"] img {
        width: auto !important;
        max-width: 100%;
        max-height: var(--lf-pane-h);
        object-fit: contain;
    }
    .st-key-brick_rows div[data-testid="stImage"] img {
        width: auto !important;
        height: calc(
            (var(--lf-pane-h) - 3.5rem - (var(--lf-rows) - 1) * var(--lf-row-gap))
            / var(--lf-rows)
        );
        max-width: 100%;
        object-fit: contain;
    }
    .side-mascot { transition: opacity 0.4s ease; }
    </style>
    """,
    unsafe_allow_html=True,
)


def upright_image(image_bytes: bytes) -> Image.Image:
    """Decode an upload the way the API sees it: EXIF orientation applied.

    A phone stores the sensor pixels and an EXIF tag saying "rotate me";
    the API's detection/predict.py applies that tag before detecting, so
    its boxes are in UPRIGHT coordinates. Drawing them on the raw pixels
    would show the photo rotated or mirrored with the boxes off-target.
    Same call, same result, on both sides.
    """
    image = Image.open(BytesIO(image_bytes))
    return ImageOps.exif_transpose(image).convert("RGB")


@st.cache_data(ttl=300, show_spinner=False)
def classifier_choices() -> list[str]:
    """Display names of the classifiers the API can serve (from /ping).

    The list lives in the API's params.py -- one source of truth -- and is
    cached for 5 minutes so the selectbox does not ping on every rerun.
    Empty when the API is unreachable or still on the old code.
    """
    try:
        response = requests.get(f"{API_URL}/ping", timeout=30)
        response.raise_for_status()
        return list(response.json()["models"]["classifiers"])
    except Exception:
        return []


def score_cards_html(inventory_coverage: float, compatibility: float) -> str:
    """The two recommendation scores as side-by-side cards.

    Both ratios come from recommender.compute_compatibility() and count
    PIECES (a 2x4 brick photographed twice counts twice), matched on
    part ID only, colour ignored:
      - inventory_coverage: pieces the set uses / pieces we photographed
        ("this set uses 50% of your bricks");
      - compatibility: pieces we have / pieces the set needs
        ("you can build 4% of this set").
    """
    cards = [
        (
            "Pièces photographiées utilisées",
            inventory_coverage,
            "Part de vos briques photographiées qui servent dans ce set. "
            "Plus c'est haut, moins il vous reste de briques inutilisées.",
        ),
        (
            "Complétion du set",
            compatibility,
            "Part du set que vous pouvez déjà construire avec vos briques. "
            "À 100 %, le set est entièrement constructible.",
        ),
    ]

    cards_html = "".join(
        f"""
        <div class="lf-score-card">
            <div class="lf-score-value">{value:.1%}</div>
            <div class="lf-score-title">{title}</div>
            <div class="lf-score-desc">{description}</div>
        </div>
        """
        for title, value, description in cards
    )

    # Indented lines would read as a Markdown code block: strip them.
    html = f"""
    <style>
    .lf-score-row {{
        display: flex;
        gap: 1rem;
        width: 100%;
        margin: 1rem 0 1.5rem 0;
    }}
    .lf-score-card {{
        flex: 1 1 0;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }}
    .lf-score-value {{
        font-size: 2.6rem;
        font-weight: 700;
        line-height: 1.1;
        color: #1f2937;
    }}
    .lf-score-title {{
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.4rem;
        color: #1f2937;
    }}
    .lf-score-desc {{
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
        line-height: 1.35;
    }}
    </style>
    <div class="lf-score-row">{cards_html}</div>
    """
    return "\n".join(line.strip() for line in html.splitlines())


def brick_rain_html() -> str:
    """It rains LEGO -- the payoff for validating every brick.

    Pure CSS: Streamlit strips <script>, but position:fixed elements with
    a keyframe animation overlay the whole page, not just one widget box.
    The four bricks (the logo's colours, app/assets/rain/) are embedded as
    data URIs -- one CSS class per brick, so each image is inlined exactly
    once however many copies fall. animation-fill-mode keeps finished
    bricks parked off-screen below the page instead of snapping back up.
    """
    import base64
    import random

    assets = sorted((Path(__file__).parent / "assets" / "rain").glob("*.png"))

    css_classes = []
    for i, path in enumerate(assets):
        uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        css_classes.append(".brick-rain .b%d { background-image: url('%s'); }" % (i, uri))

    drops = []
    for i in range(24):
        left = random.uniform(1, 94)
        size = random.randint(38, 76)
        duration = random.uniform(2.2, 4.5)
        delay = random.uniform(0, 1.8)
        rotation = random.choice((-1, 1)) * random.randint(160, 520)
        drops.append(
            '<div class="b%d" style="left:%.1fvw;width:%dpx;height:%dpx;'
            'animation-duration:%.2fs;animation-delay:%.2fs;--rotation:%ddeg;"></div>'
            % (i % len(assets), left, size, size, duration, delay, rotation)
        )

    return (
        "<style>"
        ".brick-rain div {"
        "  position: fixed; top: -90px; z-index: 9999;"
        "  background-size: contain; background-repeat: no-repeat;"
        "  pointer-events: none;"
        "  animation-name: brickfall;"
        "  animation-timing-function: linear;"
        "  animation-fill-mode: forwards;"
        "}"
        "@keyframes brickfall {"
        "  to { transform: translateY(115vh) rotate(var(--rotation)); }"
        "}"
        + "".join(css_classes)
        + "</style>"
        + '<div class="brick-rain">' + "".join(drops) + "</div>"
    )



@st.cache_data(show_spinner=False)
def side_mascots_html() -> str:
    """The two Le Wagon minifig coders, parked in the empty side gutters.

    position:fixed pins them to the viewport (guy left, girl right,
    vertically centred) outside Streamlit's centred column; below 1250px
    of window width there IS no gutter, so a media query hides them
    before they can overlap the content. pointer-events:none keeps them
    purely decorative. Cached: the data URIs never change, no point
    re-encoding two PNGs on every rerun.
    """
    import base64

    assets = Path(__file__).parent / "assets"
    uris = {}
    for name in ("coder_guy", "coder_girl"):
        data = (assets / f"{name}.png").read_bytes()
        uris[name] = "data:image/png;base64," + base64.b64encode(data).decode()

    return (
        "<style>"
        ".side-mascot {"
        "  position: fixed; z-index: 1;"
        "  width: min(19vw, 320px);"
        "  pointer-events: none;"
        "}"
        ".side-mascot.left  { left: 1.5vw;  top: 45%; transform: translateY(-50%); }"
        ".side-mascot.right { right: 1.5vw; top: 55%; transform: translateY(-50%); }"
        "@media (max-width: 1250px) { .side-mascot { display: none; } }"
        "</style>"
        + '<img class="side-mascot left" src="%s">' % uris["coder_guy"]
        + '<img class="side-mascot right" src="%s">' % uris["coder_girl"]
    )



# ============================================================
# HEADER
# ============================================================

# The logo has a transparent background with black lettering, which is
# why .streamlit/config.toml pins the light theme -- on Streamlit's dark
# theme "FITTER" would vanish. Resolved from this file, not the CWD, so
# it works from the repo root, the container and anywhere else.
# st.image left-aligns; a wide middle column centres it.
_, logo_column, _ = st.columns([1, 4, 1])
logo_column.image(
    str(Path(__file__).parent / "assets" / "logo.png"),
    use_container_width=True,
)

st.markdown(side_mascots_html(), unsafe_allow_html=True)


# ============================================================
# ANALYSE LEGO
# ============================================================

st.subheader("Identifier des pièces LEGO")

choices = classifier_choices()
selected_classifier = st.selectbox(
    "Modèle de classification",
    options=choices or ["(défaut de l'API)"],
    disabled=not choices,
    help="Le détecteur YOLO est le même ; seul le modèle qui nomme "
    "chaque brique change.",
)

uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png", "heic", "heif"],
)


if uploaded_file is not None:

    st.image(
        upright_image(uploaded_file.getvalue()),
        caption="Image à analyser",
        width=400,
    )

    if st.button("Analyser l'image"):

        try:

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            with st.spinner(
                "Détection des briques et recherche d'un set LEGO..."
            ):

                # No menu (old API) -> send nothing, the API uses its
                # default and ignores no field it never asked for.
                data = (
                    {"classifier": selected_classifier} if choices else {}
                )

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    data=data,
                    timeout=120,
                )

            if response.status_code == 200:

                # Streamlit re-runs this WHOLE script on every widget
                # interaction, and the button is only True on the click
                # itself -- so the analysis must survive in
                # session_state, or the first validation checkbox click
                # below would make the results vanish.
                st.session_state["analysis"] = response.json()
                st.session_state["analysis_image"] = uploaded_file.getvalue()

                # Fresh analysis: reset the validation state
                for key in list(st.session_state):
                    if key.startswith("brick_ok_"):
                        del st.session_state[key]

            elif response.status_code == 415:

                st.session_state.pop("analysis", None)
                st.error(
                    response.json().get(
                        "detail",
                        "Format d'image non supporté.",
                    )
                )

            else:

                st.session_state.pop("analysis", None)
                st.error(
                    f"Erreur API ({response.status_code}) : "
                    f"{response.text}"
                )

        except requests.RequestException as e:

            st.error(
                f"Impossible de contacter l'API : {e}"
            )


# ============================================================
# RÉSULTATS -- rendered from session_state so they survive the
# rerun triggered by every checkbox click
# ============================================================

if "analysis" in st.session_state:

    result = st.session_state["analysis"]

    st.success(
        f"{result['total_bricks']} pièce(s) LEGO détectée(s)"
    )

    # ------------------------------------------------
    # BOUNDING BOXES
    # ------------------------------------------------

    annotated_image = upright_image(st.session_state["analysis_image"])

    draw = ImageDraw.Draw(annotated_image)

    # Same style as app/detection/predict.py's draw_boxes():
    # labels and line widths SCALE WITH THE PHOTO (a 4000px
    # iPhone shot gets labels a room can read, a 500px render
    # gets small ones), each label sits on a filled background
    # so it never blends into a busy photo, and the colour is
    # a vivid green -- red vanished against warm-coloured
    # bricks and its bare text was unreadable.
    # Keep in sync with params.ANNOTATION_COLOR (not imported:
    # the slim Streamlit container has no dotenv, and one
    # colour is not worth widening that container for).
    annotation_color = "#00E676"

    shortest = min(annotated_image.size)
    font_size = max(16, shortest // 30)
    line_width = max(4, shortest // 300)
    pad = max(2, font_size // 5)
    font = ImageFont.load_default(size=font_size)

    for detection in result.get("detections", []):

        x1, y1, x2, y2 = detection["bbox"]

        part_id = detection["part_id"]
        confidence = detection[
            "classification_confidence"
        ]

        label = f"{part_id} {confidence:.0%}"

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=annotation_color,
            width=line_width,
        )

        # Label just above the box; just below it when the box
        # touches the top edge of the photo.
        text_width = draw.textlength(label, font=font)
        text_y = y1 - font_size - 2 * pad - line_width
        if text_y < 0:
            text_y = y2 + line_width

        draw.rectangle(
            [
                x1,
                text_y,
                x1 + text_width + 2 * pad,
                text_y + font_size + 2 * pad,
            ],
            fill=annotation_color,
        )

        draw.text(
            (x1 + pad, text_y + pad),
            label,
            fill="black",
            font=font,
        )

    # Name the classifier the API actually used (echoed in the response),
    # so a screenshot is never ambiguous about which model produced it.
    used_classifier = result.get("classifier")
    caption = "Détection YOLO + classification CNN"
    if used_classifier:
        caption += f" ({used_classifier})"

    # Photo on the left, checklist on the right, so the audience can
    # compare a label on the photo with the catalog picture next to it
    # without scrolling back and forth. The CSS at the top widens this
    # block beyond the centred page and pins the photo while the list
    # scrolls.
    photo_col, list_col = st.columns([1, 1], gap="large")

    # ------------------------------------------------
    # BOUNDING BOXES (left, sticky)
    # ------------------------------------------------

    with photo_col:

        st.subheader("Pièces détectées")

        st.image(
            annotated_image,
            caption=caption,
            use_container_width=True,
        )

        # Marker the CSS hooks (:has) to widen the block and pin this
        # column; renders as nothing. Last, not first: as the first
        # element it would push the title down a gap below the one on
        # the right.
        st.markdown('<div class="lf-breakout"></div>', unsafe_allow_html=True)

    # ------------------------------------------------
    # BRIQUES IDENTIFIÉES + validation (right)
    # ------------------------------------------------

    with list_col:

        st.subheader("Briques identifiées")

        details = result.get("inventory_details")

        if details:

            def check_all_bricks():
                """Copy the "Tout correct" box onto every per-brick box.

                Runs as the on_change callback, i.e. BEFORE the script
                reruns and before the per-brick checkboxes are
                instantiated -- the one moment Streamlit lets us write
                their session_state keys.
                """
                value = st.session_state["brick_ok_all"]
                for part in details:
                    st.session_state[f"brick_ok_{part['part_id']}"] = value

            # Outside the form on purpose: a form widget only reports on
            # submit, and this one must tick the others the instant it
            # is clicked. Same first-column width as the rows below, so
            # its box lines up with theirs; the label gets the second
            # column's room.
            header_cols = st.columns([2, 6], vertical_alignment="center")
            with header_cols[0]:
                st.checkbox(
                    "Tout correct",
                    key="brick_ok_all",
                    on_change=check_all_bricks,
                )

            # A form batches the checkboxes: clicking one changes
            # NOTHING server-side until the submit button -- no rerun,
            # no spinner, no websocket flood (unbatched, 2-3 quick
            # clicks could drop the Cloud Run session and lose the
            # analysis entirely).
            with st.form("brick_validation", border=False):

                # key= gives the container a .st-key-brick_rows class,
                # the hook the CSS uses to size the catalog pictures so
                # the rows fill the photo's height (the submit button
                # stays outside it). The row count feeds that formula.
                st.markdown(
                    f"<style>:root{{--lf-rows:{len(details)}}}</style>",
                    unsafe_allow_html=True,
                )
                rows = st.container(key="brick_rows")

                for part in details:

                    row = rows.columns([1, 1, 6], vertical_alignment="center")

                    with row[0]:
                        st.checkbox(
                            part["part_id"],
                            key=f"brick_ok_{part['part_id']}",
                            label_visibility="collapsed",
                        )

                    with row[1]:
                        image_bytes = (
                            load_part_image(part["img_url"])
                            if part.get("img_url")
                            else None
                        )
                        if image_bytes:
                            st.image(image_bytes, width=48)

                    with row[2]:
                        # One line per brick: the part ID first and
                        # big (what the audience just saw on the boxes
                        # in the photo), the catalog name as small grey
                        # print after it. One element, not two, keeps
                        # the row as short as the image.
                        name = part.get("name") or ""
                        st.markdown(
                            f"**{part['part_id']}** — {part['count']} × "
                            f'<span style="color:#6b7280;font-size:0.9rem">'
                            f"{name}</span>",
                            unsafe_allow_html=True,
                        )

                submitted = st.form_submit_button("Valider les briques")

            if submitted:

                checked = [
                    part
                    for part in details
                    if st.session_state.get(f"brick_ok_{part['part_id']}")
                ]

                if len(checked) == len(details):
                    st.success("Toutes les briques sont validées !")
                    st.markdown(
                        brick_rain_html(),
                        unsafe_allow_html=True,
                    )
                else:
                    st.info(
                        f"{len(checked)}/{len(details)} briques validées."
                    )

        else:
            # API without part details (no Rebrickable key, or an
            # older deployment): raw counts beat nothing.
            st.json(result["inventory"])

    # The widened block runs under the side mascots, so they fade out
    # while it is on screen and come back once the page scrolls on to
    # the recommended set. Streamlit strips <script> from st.markdown,
    # but a components.html iframe runs it and can reach the parent
    # page. One poller per page (guarded on window.parent), 200ms --
    # cheaper than wiring a scroll listener to Streamlit's own scroll
    # container, whose selector changes between versions.
    components.html(
        """
        <script>
        const win = window.parent;
        if (!win.__lfMascotWatch) {
            win.__lfMascotWatch = setInterval(() => {
                const doc = win.document;
                const marker = doc.querySelector(".lf-breakout");
                const block = marker
                    && marker.closest('[data-testid="stHorizontalBlock"]');
                let hide = false;
                if (block) {
                    const r = block.getBoundingClientRect();
                    hide = r.top < win.innerHeight && r.bottom > 0;
                }
                doc.querySelectorAll(".side-mascot").forEach((m) => {
                    m.style.opacity = hide ? "0" : "1";
                });
            }, 200);
        }
        </script>
        """,
        height=0,
    )

    # ------------------------------------------------
    # SET RECOMMANDÉ
    # ------------------------------------------------

    recommended_set = result.get(
        "recommended_set"
    )

    if recommended_set:

        st.divider()

        st.subheader("Set LEGO recommandé")

        st.write(
            f"### {recommended_set['name']}"
        )

        st.write(
            f"Set : {recommended_set['set_num']}"
        )

        if recommended_set.get("set_img_url"):

            st.image(
                recommended_set["set_img_url"],
                width=400,
            )

        st.markdown(
            score_cards_html(
                recommended_set["inventory_coverage"],
                recommended_set["compatibility"],
            ),
            unsafe_allow_html=True,
        )

        if recommended_set["buildable"]:

            st.success(
                "Ce set est entièrement constructible "
                "avec les pièces détectées."
            )

        else:

            st.info(
                "C'est le set le plus compatible "
                "avec les pièces détectées."
            )

    else:

        st.warning(
            "Aucun set LEGO compatible trouvé."
        )
