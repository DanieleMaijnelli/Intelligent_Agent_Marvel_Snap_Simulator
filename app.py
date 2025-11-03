from flask import *
from gameManager import GameState
import json, csv, os, io
import traceback
import csv
from io import StringIO
from flask import Response

game = GameState()
game.gameStart()

app = Flask(__name__)
MOVE_DATA_PATH = "matchlogs/move-data.json"
GAME_DATA_PATH = "matchlogs/game-data.json"


def load_json(filename):
    with open(f"matchlogs/{filename}", "r") as file:
        return json.load(file)


def normalize_name(name):
    return name.replace(" ", "").capitalize()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/game/ally')
def gameAlly():
    print(game.status["allyhand"])
    return render_template('allygame.html', status=game.status, locations=game.locationList.values())


@app.route('/check_game_end', methods=['GET'])
def checkGameEnded():
    winner = game.passStatus['winner']
    print(game.passStatus)
    print('winner is ', winner)
    return jsonify(winner)


@app.route('/game/ally/playcard', methods=['POST'])
def chooseLocationAlly():
    inputUnit = int(request.form['index'])
    return render_template('playCardAllies.html', status=game.status, locations=game.locationList.values(),
                           card=game.status["allyhand"][inputUnit], cardnum=inputUnit)


@app.route('/game/ally/playcard/<locationnum>', methods=['POST'])
def playCardAlly(locationnum):
    locationToAdd = int(locationnum) + 1
    inputUnit = int(request.form["card"])
    was_added = False
    if not game.status['allypass']:
        try:
            if game.status["allyenergy"] < game.status["allyhand"][inputUnit].cur_cost:
                print("not enough energy")
            else:
                was_added = game.addUnit(game.status["allyhand"][inputUnit], True, locationToAdd)
            if was_added:
                game.status["allyenergy"] -= game.status["allyhand"][inputUnit].cur_cost
                del game.status["allyhand"][inputUnit]
        except Exception as e:
            print("error")
            print(e)
    return redirect(url_for('gameAlly'))


@app.route('/game/enemy')
def gameEnemy():
    return render_template('enemygame.html', status=game.status, locations=game.locationList.values())


@app.route('/game/enemy/playcard', methods=['POST'])
def chooseLocationEnemy():
    inputUnit = int(request.form['index'])
    return render_template('playCardEnemies.html', status=game.status, locations=game.locationList.values(),
                           card=game.status["enemyhand"][inputUnit], cardnum=inputUnit)


@app.route('/game/enemy/playcard/<locationnum>', methods=['POST'])
def playCardEnemy(locationnum):
    locationToAdd = int(locationnum) + 1
    inputUnit = int(request.form["card"])
    was_added = False
    if not game.status['enemypass']:
        try:
            if game.status["enemyenergy"] < game.status["enemyhand"][inputUnit].cur_cost:
                print("not enough energy")
            else:
                was_added = game.addUnit(game.status["enemyhand"][inputUnit], False, locationToAdd)
            if was_added:
                game.status["enemyenergy"] -= game.status["enemyhand"][inputUnit].cur_cost
                del game.status["enemyhand"][inputUnit]
        except Exception as e:
            print("error")
            print(e)
    return redirect(url_for('gameEnemy'))


@app.route('/game/<allyorenemy>/moveUnit', methods=['POST'])
def moveUnit(allyorenemy):
    cardnum = int(request.form["card"])
    fromlocation = int(request.form["location_num"]) + 1
    location = "location" + str(fromlocation)
    if allyorenemy == "ally":
        card = game.locationlist[location].allies[cardnum]
        return render_template('moveCardAllies.html', card=card)
    elif allyorenemy == "enemy":
        card = game.locationlist[location].enemies[cardnum]
        return render_template('moveCardEnemies.html', card=card)


@app.route('/game/<allyorenemy>/pass', methods=['POST'])
def passTurn(allyorenemy):
    if allyorenemy == "ally":
        game.status["allypass"] = True
    elif allyorenemy == "enemy":
        game.status["enemypass"] = True
    print("pass")

    return jsonify(success=True)


@app.route('/check_turn/<allyorenemy>')
def check_turn(allyorenemy):
    print(allyorenemy)
    if game.status["allypass"] and game.status["enemypass"]:
        if allyorenemy == "ally":
            game.status["endofturncounterally"] = 1
            game.passStatus['turnpassally'] = True

        elif allyorenemy == "enemy":
            game.status["endofturncounterenemy"] = 1
            game.passStatus['turnpassenemy'] = True

        if game.status["endofturncounterally"] == 1 and game.status["endofturncounterenemy"] == 1:
            print("End turn triggered.")
            game.turnEnd(False)
            print(game.passStatus)
            if game.passStatus['retreatally'] and game.passStatus['retreatenemy']:
                game.passStatus['winner'] = "Tie"
            elif game.passStatus['retreatally']:
                game.passStatus['winner'] = "Enemy"
                game.status['cubes'] /= 2
                game.endGame()
            elif game.passStatus['retreatenemy']:
                game.passStatus['winner'] = "Ally"
                game.status['cubes'] /= 2
                game.endGame()
            elif game.status["turncounter"] > game.status["maxturns"]:
                game.endGame()
                game.passStatus['winner'] = game.checkWinner()
            print(game.passStatus)
    return game.passStatus


@app.route('/game/<allyorenemy>/endgame', methods=['GET'])
def endGame(allyorenemy):
    if allyorenemy == "ally":
        return render_template('endgameally.html', status=game.status, locations=game.locationList.values(),
                               passStatus=game.passStatus)
    elif allyorenemy == "enemy":
        return render_template('endgameenemy.html', status=game.status, locations=game.locationList.values(),
                               passStatus=game.passStatus)


@app.route("/game/startofturn", methods=['GET'])
def startOfTurn():
    print("start of turn")
    game.startOfTurn()
    game.status['allypass'] = False
    game.status['enemypass'] = False
    game.status['endofturncounterally'] = 0
    game.status['endofturncounterenemy'] = 0
    game.passStatus = {
        'turnpassally': game.status['allypass'],
        'turnpassenemy': game.status['enemypass'],
        'winner': game.passStatus['winner'],
        'retreatally': game.passStatus['retreatally'],
        'retreatenemy': game.passStatus['retreatenemy'],
        'turnend': False
    }
    return "ok"


@app.route("/game/reset", methods=['GET'])
def resetGame():
    print("reset")
    game.reset()
    return redirect((url_for('home')))


@app.route("/game/<allyorenemy>/undoActions", methods=['POST'])
def undoActions(allyorenemy):
    if allyorenemy == "ally":
        game.status["allyenergy"] += game.undoActions(True, game.status["allyhand"])
    elif allyorenemy == "enemy":
        game.status["enemyenergy"] += game.undoActions(False, game.status["enemyhand"])

    if allyorenemy == "ally":
        return redirect(url_for('gameAlly'))
    elif allyorenemy == "enemy":
        return redirect(url_for('gameEnemy'))


@app.route("/game/<allyorenemy>/snap", methods=['POST'])
def snap(allyorenemy):
    if allyorenemy == "ally":
        game.snap(True)
        return redirect(url_for('gameAlly'))
    elif allyorenemy == "enemy":
        game.snap(False)
        return redirect(url_for('gameEnemy'))


'''@app.route("/game/<allyorenemy>/retreat", methods=['POST'])
def retreat(allyorenemy):
    if allyorenemy == "ally":
        game.passStatus['retreatally'] = True
        game.passStatus['turnpassally'] = True
        return redirect(url_for('gameAlly'))
    elif allyorenemy == "enemy":
        game.passStatus['retreatenemy'] = True
        game.passStatus['turnpassenemy'] = True
        return redirect(url_for('gameEnemy'))'''

@app.route("/game/<allyorenemy>/retreat", methods=['POST'])
def retreat(allyorenemy):
    # Flagga il retreat e chi ha vinto
    if allyorenemy == "ally":
        game.passStatus['retreatally'] = True
        game.passStatus['winner'] = "Enemy"
    elif allyorenemy == "enemy":
        game.passStatus['retreatenemy'] = True
        game.passStatus['winner'] = "Ally"

    if(game.passStatus['retreatally'] and game.passStatus['retreatenemy']):
        game.passStatus['winner'] = "Tie"
    # Esponi i cubes finali nel payload usato dal template
    game.passStatus['cubes'] = game.status['cubes']

    # Vai direttamente alla pagina di endgame (così nulla ti sovrascrive lo stato)
    return redirect(url_for('endGame', allyorenemy=allyorenemy))



@app.route("/game/<allyorenemy>/movecard", methods=['POST'])
def chooseLocationForMoveCard(allyorenemy):
    locationNum = int(request.form["locationNum"])
    location = game.locationList["location" + str(locationNum)]
    cardNum = int(request.form["card"])
    if allyorenemy == "ally":
        card = location.allies[cardNum]
        return render_template('moveCardsAllies.html', status=game.status, locations=game.locationList.values(),
                               passStatus=game.passStatus, card=card, cardnum=cardNum)
    elif allyorenemy == "enemy":
        card = location.enemies[cardNum]
        return render_template('moveCardsEnemies.html', status=game.status, locations=game.locationList.values(),
                               passStatus=game.passStatus, card=card, cardnum=cardNum)


@app.route("/game/<allyorenemy>/movecard/<locationnum>", methods=['POST'])
def confirmMove(allyorenemy, locationnum):
    originalLocation = int(request.form["oglocation"])
    locationnum = int(locationnum) + 1
    if allyorenemy == "ally":
        card = game.locationList["location" + str(originalLocation)].allies[int(request.form["card"])]
    elif allyorenemy == "enemy":
        card = game.locationList["location" + str(originalLocation)].enemies[int(request.form["card"])]
    newlocation = game.locationList["location" + str(locationnum)]
    result = game.moveSelection(card, newlocation)
    if allyorenemy == "ally":
        return redirect(url_for('gameAlly'))
    elif allyorenemy == "enemy":
        return redirect(url_for('gameEnemy'))


@app.route("/data/<version>/<cardname>", methods=['GET'])
def getCardData(version, cardname):
    try:
        MOVE_DATA_PATH = f"matchlogs/moves/move-{version}-data.json"
        GAME_DATA_PATH = f"matchlogs/games/game-{version}-data.json"
        with open(MOVE_DATA_PATH, 'r') as file:
            all_moves = json.load(file)
        with open(GAME_DATA_PATH, 'r') as f_games:
            all_games = json.load(f_games)
    except Exception as e:
        return jsonify({"error": f"Errore nel caricamento dati: {str(e)}"}), 500

    winners_by_game = {game["game_id"]: game.get("winner") for game in all_games}
    filtered_moves = []
    for move in all_moves:
        if move.get("card_played", "").strip().lower() == cardname.strip().lower():
            game_id = move.get("game_id")
            winner = winners_by_game.get(game_id)
            move_with_winner = move.copy()
            move_with_winner["winner"] = winner
            filtered_moves.append(move_with_winner)
            print(move)
    return render_template("data/card-data.html", moves=filtered_moves, cardname=cardname, version=version)


@app.route("/data/<version>/games", methods=['GET'])
def getGamesData(version):
    try:
        GAME_DATA_PATH = f"games/game-{version}-data.json"
        games = load_json(GAME_DATA_PATH)
        return render_template("data/game-data.html", games=games, version=version)
    except Exception as e:
        print(e)
        return render_template("data/game-data.html", games=[])


@app.route("/data/<version>/moves", methods=['GET'])
def getMovesData(version):
    try:
        MOVE_DATA_PATH = f"moves/move-{version}-data.json"
        moves = load_json(MOVE_DATA_PATH)
        return render_template("data/allmoves.html", moves=moves, version=version)
    except Exception as e:
        print(e)
        return render_template("data/allmoves.html", moves=[], version=version)


@app.route("/data/games/<version>/<game_id>/export/csv", methods=['GET'])
def export_game_moves_csv(version, game_id):
    try:
        MOVE_DATA_PATH = f"matchlogs/moves/move-{version}-data.json"
        with open(MOVE_DATA_PATH, "r") as f:
            moves = json.load(f)
        game_moves = [m for m in moves if m["game_id"] == game_id]

        if not game_moves:
            abort(404)
        csv_output = StringIO()
        writer = csv.writer(csv_output)

        writer.writerow(["Turn", "Player", "Card Played", "Location Name", "Position"])

        for move in game_moves:
            writer.writerow([
                move.get("turn", ""),
                move.get("player", ""),
                move.get("card_played", ""),
                move.get("location", {}).get("name", ""),
                move.get("location", {}).get("position", "")
            ])

        response = Response(csv_output.getvalue(), mimetype="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename={version}_game_{game_id}_moves.csv"
        return response

    except Exception as e:
        print(f"Errore nell'esportazione CSV: {e}")
        traceback.print_exc()
        abort(500)


@app.route("/data/<version>/games/<game_id>/export/json", methods=["GET"])
def export_game_moves_json(version, game_id):
    try:
        MOVE_DATA_PATH = f"matchlogs/moves/move-{version}-data.json"
        with open(MOVE_DATA_PATH, "r") as f:
            moves = json.load(f)
        game_moves = [m for m in moves if m["game_id"] == game_id]

        if not game_moves:
            abort(404)

        response = make_response(json.dumps(game_moves, indent=2))
        response.headers["Content-Type"] = "application/json"
        response.headers["Content-Disposition"] = f"attachment; filename=game_{version}_{game_id}_moves.json"
        return response

    except Exception as e:
        print(f"Errore nell'esportazione JSON: {e}")
        traceback.print_exc()
        abort(500)


@app.route("/data/<version>/games/<game_id>", methods=['GET'])
def getGameById(version, game_id):
    try:
        # Carica le mosse
        MOVE_DATA_PATH = f"matchlogs/moves/move-{version}-data.json"
        GAME_DATA_PATH = f"matchlogs/games/game-{version}-data.json"
        with open(MOVE_DATA_PATH, "r") as f:
            moves = json.load(f)
        game_moves = [m for m in moves if m["game_id"] == game_id]
        with open(GAME_DATA_PATH, "r") as f:
            games = json.load(f)
        game = next((g for g in games if g["game_id"] == game_id), None)
        if not game_moves:
            abort(404)
        return render_template("data/moves-by-game.html", moves=game_moves, game=game, version=version)
    except Exception as e:
        print(f"Errore nel caricamento: {e}")
        traceback.print_exc()
        abort(500)


@app.route("/data/<version>/moves/export", methods=['GET'])
def export_moves_csv(version):
    try:
        MOVE_DATA_PATH = f"matchlogs/moves/move-{version}-data.json"
        with open(MOVE_DATA_PATH, "r") as f:
            data = json.load(f)

        if not data:
            return "No data available", 404

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)

        response = Response(output.getvalue(), mimetype='text/csv')
        filename = f"move-{version}-data.csv"
        response.headers.set("Content-Disposition", "attachment", filename=filename)
        return response

    except FileNotFoundError:
        return "File not found", 404


@app.route("/data/<version>/moves.json")
def send_moves_json(version):
    filepath = f"matchlogs/moves/move-{version}-data.json"
    return send_file(filepath, mimetype="application/json")


@app.route("/data/<version>/games.json")
def send_games_json(version):
    filepath = f"matchlogs/games/game-{version}-data.json"
    return send_file(filepath, mimetype="application/json")


@app.route("/data/<version>/games/export", methods=['GET'])
def export_games_csv(version):
    try:
        GAME_DATA_PATH = f"matchlogs/games/game-{version}-data.json"
        with open(GAME_DATA_PATH, "r") as f:
            data = json.load(f)
        if not data:
            return "No data available", 404

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)

        response = Response(output.getvalue(), mimetype='text/csv')
        filename = f"game-{version}-data.csv"
        response.headers.set("Content-Disposition", "attachment", filename=filename)
        return response

    except FileNotFoundError:
        return "File not found", 404


@app.route("/metadata/<version>/moves")
def view_moves_metadata(version):
    with open(f"matchlogs/moves/move-{version}-metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return render_template("data/metadata.html",
                           title=metadata["dc:title"],
                           description=metadata["dc:description"],
                           creator=metadata["dc:creator"],
                           rights=metadata["dc:rights"],
                           identifier=metadata["dc:identifier"],
                           raw_metadata=metadata)


@app.route("/metadata/<version>/games")
def view_games_metadata(version):
    with open(f"matchlogs/games/game-{version}-metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return render_template("data/metadata.html",
                           title=metadata["dc:title"],
                           description=metadata["dc:description"],
                           creator=metadata["dc:creator"],
                           rights=metadata["dc:rights"],
                           identifier=metadata["dc:identifier"],
                           raw_metadata=metadata)


if __name__ == "__main__":
    app.run(debug=True)
