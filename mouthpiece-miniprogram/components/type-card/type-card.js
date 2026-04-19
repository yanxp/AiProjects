Component({
  properties: {
    type: { type: Object, value: {} }
  },
  methods: {
    onTap() {
      const code = this.data.type.code
      wx.navigateTo({ url: `/pages/type-detail/type-detail?code=${code}` })
    }
  }
})
