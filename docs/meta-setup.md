# Connect Facebook and Instagram

Arsu Clip uses one Facebook Page and the professional Instagram account linked to that Page. Connect an account you control and verify the Page name and Instagram username shown in the studio.

1. In [Meta Business Suite](https://business.facebook.com/), check that the Facebook Page and the Instagram Business or Creator account are linked. Confirm that your Facebook user can publish content to the Page.
2. Open [Meta for Developers](https://developers.facebook.com/apps/). Use your own Meta app with Facebook Login for Business and the Instagram API with Facebook login configured. For internal use, use an account with access to the app and the Page. Broader use may require Meta App Review.
3. In [Graph API Explorer](https://developers.facebook.com/tools/explorer/), select your Meta app and generate a **User Token**. Grant `pages_show_list`, `pages_read_engagement`, `pages_manage_posts`, `instagram_basic`, and `instagram_content_publish`. Select the Page and Instagram account you intend to use. For regular use, extend the token through Meta's [Access Token Debugger](https://developers.facebook.com/tools/debug/accesstoken/) when available.
4. In Arsu Clip, open **Integrations → Facebook & Instagram**. Enter the Facebook Page ID, the Instagram username, and your Meta user access token. Click **Connect accounts**, then **Test connection**. Enter tokens only in your own local studio.
5. Open a finished clip and select **Post to Facebook** or **Post to Instagram**. The clip card shows the result. For Instagram auto-posting or scheduling, opt in when creating the project and keep Arsu Clip running when posts are due.

For Facebook, use 9:16 clips up to 90 seconds. If Meta rejects an Instagram upload, check the error in the clip card. You can use Meta Business Suite to post a rejected clip manually. If Arsu Clip reports that it could not confirm a post, check **Published** in Meta Business Suite before trying again.

The verified connection is saved locally as `data/meta_token.json` by default, or in your custom data folder. If access expires or is revoked, repeat the connection step with a fresh token. For more on media requirements and permissions, see the [Facebook Reels guide](https://developers.facebook.com/documentation/video-api/guides/reels-publishing) and [Instagram publishing guide](https://developers.facebook.com/documentation/instagram-platform/content-publishing).
