if __name__ == "__main__":

    from whila.appify import init_modal_instance

    app = init_modal_instance()

    with app.run():
        from whila.appify import ModalApp

        modal_app = ModalApp()
        modal_app.launch()
