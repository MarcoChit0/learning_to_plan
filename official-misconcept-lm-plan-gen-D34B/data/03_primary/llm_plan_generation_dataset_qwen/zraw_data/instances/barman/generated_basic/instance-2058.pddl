(define (problem prob)
 (:domain barman)
 (:objects 
      shaker280 - shaker
      left right - hand
      shot395 - shot
      ingredient262 ingredient89 ingredient145 ingredient132 - ingredient
      cocktail486 - cocktail
      dispenser131 dispenser241 dispenser300 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker280)
  (ontable shot395)
  (dispenses dispenser131 ingredient262)
  (dispenses dispenser241 ingredient89)
  (dispenses dispenser300 ingredient145)
  (dispenses dispenser370 ingredient132)
  (clean shaker280)
  (clean shot395)
  (empty shaker280)
  (empty shot395)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker280 l0)
  (shaker-level shaker280 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail486 ingredient89)
  (cocktail-part2 cocktail486 ingredient262)
)
 (:goal
  (and
      (contains shot395 cocktail486)
)))
