(define (problem prob)
 (:domain barman)
 (:objects 
      shaker6 - shaker
      left right - hand
      shot55 shot223 - shot
      ingredient311 ingredient488 ingredient215 - ingredient
      cocktail431 - cocktail
      dispenser393 dispenser113 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker6)
  (ontable shot55)
  (ontable shot223)
  (dispenses dispenser393 ingredient311)
  (dispenses dispenser113 ingredient488)
  (dispenses dispenser240 ingredient215)
  (clean shaker6)
  (clean shot55)
  (clean shot223)
  (empty shaker6)
  (empty shot55)
  (empty shot223)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker6 l0)
  (shaker-level shaker6 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail431 ingredient311)
  (cocktail-part2 cocktail431 ingredient488)
)
 (:goal
  (and
      (contains shot55 cocktail431)
)))
