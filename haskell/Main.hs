module Main where

import System.Process (readProcess)
import System.Directory (createDirectoryIfMissing)
import System.IO (hFlush, stdout, hSetBuffering, BufferMode(NoBuffering))
import Data.Char (ord, chr)
import Data.List (intercalate)

-- Importaciones de Red corregidas
import Network.Socket hiding (send, recv)
import Network.Socket.ByteString (recv, sendAll)
import qualified Data.ByteString.Char8 as BS

type FEN = String

initialFEN :: FEN
initialFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"

-- Definimos el puerto explícitamente
port :: PortNumber
port = 65432

main :: IO ()
main = do
  hSetBuffering stdout NoBuffering
  putStrLn "================================================="
  putStrLn "  AJEDREZ MOTORIZADO: PROLOG + HASKELL + VISION  "
  putStrLn "================================================="
  putStrLn $ "Esperando conexión del módulo de visión en puerto " ++ show port ++ "..."
  
  createDirectoryIfMissing True "web"
  
  -- Configuración del Socket Servidor
  -- 1. Creamos el socket TCP/IPv4
  sock <- socket AF_INET Stream 0
  
  -- 2. Permitimos reutilizar el puerto inmediatamente (evita error "Address already in use")
  setSocketOption sock ReuseAddr 1
  
  -- 3. Bind (Enlace) corregido: Usamos tupleToHostAddress en lugar de iNADDR_ANY
  bind sock (SockAddrInet port (tupleToHostAddress (0,0,0,0)))
  
  -- 4. Escuchar conexiones (cola de 2)
  listen sock 2
  
  -- Iniciar bucle de juego
  gameLoop sock initialFEN []

-- Bucle modificado para aceptar conexiones TCP en lugar de stdin
gameLoop :: Socket -> FEN -> [String] -> IO ()
gameLoop sock fen history = do
  writeBoardHtml "web/board.html" fen Nothing
  putStrLn $ "\n[Estado] FEN: " ++ fen
  putStrLn "[Esperando] Realiza un movimiento frente a la cámara..."
  
  -- Bloquea hasta que Python envía un comando
  (conn, _) <- accept sock
  msgBS <- recv conn 1024
  let cmd = BS.unpack msgBS
  close conn -- Cerramos la conexión tras recibir el comando (stateless per move)
  
  if null cmd 
    then gameLoop sock fen history -- Conexión vacía, ignorar
    else do
      processCommand sock fen history cmd

processCommand :: Socket -> FEN -> [String] -> String -> IO ()
processCommand sock fen history cmd = do
  case cmd of
    ":new" -> gameLoop sock initialFEN []
    _ -> do
      putStrLn $ "[Recibido] Jugada detectada: " ++ cmd
      newFen <- prologApply fen cmd
      
      if newFen == "illegal"
        then do
          putStrLn ">> MOVIMIENTO ILEGAL DETECTADO <<"
          gameLoop sock fen history
        else do
          let history1 = history ++ [cmd]
          writeBoardHtml "web/board.html" newFen (Just cmd)
          
          -- Turno del Sistema (IA)
          aiMove <- prologBestMove newFen
          if aiMove == "none"
            then handleGameOver sock newFen cmd
            else do
              putStrLn $ "[Sistema] IA responde: " ++ aiMove
              fen2 <- prologApply newFen aiMove
              writeBoardHtml "web/board.html" fen2 (Just aiMove)
              gameLoop sock fen2 (history1 ++ [aiMove])

handleGameOver :: Socket -> FEN -> String -> IO ()
handleGameOver sock fen lastMove = do
  st <- prologStatus fen
  putStrLn $ "Juego terminado. Estado: " ++ st
  case st of
    "checkmate" -> do 
         putStrLn "Mate. Aprendiendo..." 
         _ <- prologLearnSimple "system_loss"
         return ()
    "stalemate" -> do
         _ <- prologLearnSimple "draw"
         return ()
    _ -> return ()
  writeBoardHtml "web/board.html" fen (Just lastMove)
  -- Reiniciar o salir
  gameLoop sock initialFEN []

-- =========================================================
--  Puente con Prolog (Sin cambios significativos)
-- =========================================================

prologBestMove :: FEN -> IO String
prologBestMove fen = do
  let goal = "cli_best_move('" ++ esc fen ++ "')"
  trim <$> callSwipl goal

prologApply :: FEN -> String -> IO String
prologApply fen uci = do
  let goal = "cli_apply_move('" ++ esc fen ++ "','" ++ esc uci ++ "')"
  trim <$> callSwipl goal

prologStatus :: FEN -> IO String
prologStatus fen = do
  let goal = "cli_status('" ++ esc fen ++ "')"
  trim <$> callSwipl goal

prologLearnSimple :: String -> IO String
prologLearnSimple outcome = do
  let goal = "learn_simple(" ++ outcome ++ ")"
  trim <$> callSwipl goal

callSwipl :: String -> IO String
callSwipl goal = readProcess "swipl" ["-q","-s","prolog/chess.pl","-g",goal,"-t","halt"] ""

esc :: String -> String
esc = concatMap escapeChar
  where
    escapeChar '\'' = "\\'"
    escapeChar c    = [c]

trim :: String -> String
trim = f . f
  where f = reverse . dropWhile (`elem` ['\n','\r',' '])

-- =========================================================
--  Generación HTML (Utiliza la lógica previa)
-- =========================================================
writeBoardHtml :: FilePath -> FEN -> Maybe String -> IO ()
writeBoardHtml path fen mLast = do
  let board = fenToBoard fen
  status <- prologStatus fen
  let statusMsg = gameOverMessage fen status
  let html = renderHtml board mLast statusMsg
  writeFile path html

fenToBoard :: FEN -> [String]
fenToBoard fen = expand (takeWhile (/= ' ') fen)
  where
    expand s = concatMap rankToCells (splitOn '/' s)
    rankToCells rs = go rs []
    go [] acc = reverse acc
    go (c:cs) acc
      | c >= '1' && c <= '8' = let n = ord c - ord '0' in go cs (replicate n "" ++ acc)
      | otherwise = go cs ([ [c] ] ++ acc)

splitOn :: Char -> String -> [String]
splitOn d s = case break (== d) s of
  (a, _ : b) -> a : splitOn d b
  (a, [])    -> [a]

renderHtml :: [String] -> Maybe String -> Maybe String -> String
renderHtml cells mLast mStatus = unlines
  [ "<!doctype html>"
  , "<html lang=\"es\">"
  , "<head>"
  , "  <meta charset=\"utf-8\" />"
  , "  <meta http-equiv=\"refresh\" content=\"1\" />" -- Refresco más rápido para visión
  , "  <title>Ajedrez IA Vision</title>"
  , "  <style>"
  , "    body{font-family:system-ui,sans-serif;background:#111;color:#eee;display:grid;place-items:center;}"
  , "    table{border-collapse:collapse;margin:12px 0;box-shadow:0 0 20px rgba(0,0,0,0.5);}"
  , "    td{width:60px;height:60px;text-align:center;font-size:40px;cursor:default;}"
  , "    .light{background:#eeeed2;color:#000;} .dark{background:#769656;color:#000;}"
  , "    .legend{font-size:12px;color:#888;}"
  , "    .banner{padding:10px;margin:10px;border-radius:4px;text-align:center;width:100%;}"
  , "    .banner.warn{background:#d32f2f;}"
  , "  </style>"
  , "</head>"
  , "<body>"
  , maybe "" (\m -> "<div class=\"banner warn\">" ++ m ++ "</div>") mStatus
  , boardTable cells
  , maybe "" (\m -> "<p>Último movimiento: <strong>" ++ m ++ "</strong></p>") mLast
  , "</body></html>"
  ]

boardTable :: [String] -> String
boardTable cells =
  let fileHeader = "<tr><td class='legend'></td>" ++ concat [ fileCell c | c <- [0..7] ] ++ "<td class='legend'></td></tr>"
      rows = [ rowWithLegends r | r <- [0..7] ]
  in "<table>" ++ fileHeader ++ concat rows ++ fileHeader ++ "</table>"
  where
    fileCell c = "<td class='legend'>" ++ [chr (ord 'a' + c)] ++ "</td>"
    rankLabel r = show (8 - r)
    rowWithLegends r =
      "<tr>"
      ++ "<td class='legend'>" ++ rankLabel r ++ "</td>"
      ++ concat [ cellAt r c | c <- [0..7] ]
      ++ "<td class='legend'>" ++ rankLabel r ++ "</td>"
      ++ "</tr>"
    cellAt r c = let i = r*8 + c in tdClass r c (cellContent (cells !! i))

tdClass :: Int -> Int -> String -> String
tdClass r c inner =
  let isLight = (r + c) `mod` 2 == 0
      cls = if isLight then "light" else "dark"
  in "<td class='" ++ cls ++ "'>" ++ inner ++ "</td>"

cellContent :: String -> String
cellContent s = case s of
  ""   -> ""
  "P"  -> "\x2659"; "N"  -> "\x2658"; "B"  -> "\x2657"; "R"  -> "\x2656"; "Q"  -> "\x2655"; "K"  -> "\x2654"
  "p"  -> "\x265F"; "n"  -> "\x265E"; "b"  -> "\x265D"; "r"  -> "\x265C"; "q"  -> "\x265B"; "k"  -> "\x265A"
  _    -> "?"

gameOverMessage :: FEN -> String -> Maybe String
gameOverMessage fen status =
  case status of
    "checkmate" -> Just "JAQUE MATE"
    "stalemate" -> Just "TABLAS (Ahogado)"
    _           -> Nothing

sideToMove :: FEN -> String
sideToMove fen = case words fen of (_:side:_) -> side; _ -> "w"