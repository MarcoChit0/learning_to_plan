(define (problem prob)
 (:domain barman)
 (:objects 
      shaker488 - shaker
      left right - hand
      shot404 shot239 - shot
      ingredient191 ingredient466 ingredient250 ingredient497 - ingredient
      cocktail47 - cocktail
      dispenser241 dispenser270 dispenser313 dispenser135 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker488)
  (ontable shot404)
  (ontable shot239)
  (dispenses dispenser241 ingredient191)
  (dispenses dispenser270 ingredient466)
  (dispenses dispenser313 ingredient250)
  (dispenses dispenser135 ingredient497)
  (clean shaker488)
  (clean shot404)
  (clean shot239)
  (empty shaker488)
  (empty shot404)
  (empty shot239)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker488 l0)
  (shaker-level shaker488 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail47 ingredient191)
  (cocktail-part2 cocktail47 ingredient466)
)
 (:goal
  (and
      (contains shot404 cocktail47)
)))
