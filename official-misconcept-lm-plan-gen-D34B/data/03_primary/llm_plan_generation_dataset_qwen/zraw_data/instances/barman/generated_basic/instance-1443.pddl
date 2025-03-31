(define (problem prob)
 (:domain barman)
 (:objects 
      shaker353 - shaker
      left right - hand
      shot13 - shot
      ingredient153 ingredient46 ingredient493 - ingredient
      cocktail107 - cocktail
      dispenser309 dispenser296 dispenser13 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker353)
  (ontable shot13)
  (dispenses dispenser309 ingredient153)
  (dispenses dispenser296 ingredient46)
  (dispenses dispenser13 ingredient493)
  (clean shaker353)
  (clean shot13)
  (empty shaker353)
  (empty shot13)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker353 l0)
  (shaker-level shaker353 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail107 ingredient46)
  (cocktail-part2 cocktail107 ingredient493)
)
 (:goal
  (and
      (contains shot13 cocktail107)
)))
