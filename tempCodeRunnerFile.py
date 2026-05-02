
            self.mode = new_mode
            self.analyzer.set_mode(self.mode)
            self.mode_badge.set_mode(self.mode, MODE_COLORS[self.mode])
            self.caption_pill.set_text(MODE_CAPTIONS[self.mode])
            self.caption_pill.set_color(MODE_COLORS[self.mode])
            self.flash.trigger(self.mode, MODE_COLORS[self.mode])