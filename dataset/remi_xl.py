import midi2corpus
import corpus2events
import events2words
import compile


def main():
    path_root = "./data"
    print("[MAIN] midi -> corpus >>>>>")
    midi2corpus(path_root)
    print("[MAIN] corpus -> events >>>>>")
    corpus2events(path_root)
    print("[MAIN] events -> words >>>>>")
    events2words(path_root)
    print("[MAIN] words -> npz >>>>>")
    compile(path_root)
    print("[MAIN] finished! <<<<<")


if __name__ == "__main__":
    main()
