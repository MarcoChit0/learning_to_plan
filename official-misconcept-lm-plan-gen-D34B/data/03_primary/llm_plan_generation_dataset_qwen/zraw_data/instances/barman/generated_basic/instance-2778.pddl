(define (problem prob)
 (:domain barman)
 (:objects 
      shaker248 - shaker
      left right - hand
      shot231 - shot
      ingredient406 ingredient336 ingredient109 ingredient242 - ingredient
      cocktail164 - cocktail
      dispenser442 dispenser138 dispenser7 dispenser493 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker248)
  (ontable shot231)
  (dispenses dispenser442 ingredient406)
  (dispenses dispenser138 ingredient336)
  (dispenses dispenser7 ingredient109)
  (dispenses dispenser493 ingredient242)
  (clean shaker248)
  (clean shot231)
  (empty shaker248)
  (empty shot231)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker248 l0)
  (shaker-level shaker248 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient242)
  (cocktail-part2 cocktail164 ingredient336)
)
 (:goal
  (and
      (contains shot231 cocktail164)
)))
