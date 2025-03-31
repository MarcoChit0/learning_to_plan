(define (problem prob)
 (:domain barman)
 (:objects 
      shaker304 - shaker
      left right - hand
      shot364 - shot
      ingredient120 ingredient426 ingredient342 ingredient265 - ingredient
      cocktail80 - cocktail
      dispenser13 dispenser62 dispenser127 dispenser221 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker304)
  (ontable shot364)
  (dispenses dispenser13 ingredient120)
  (dispenses dispenser62 ingredient426)
  (dispenses dispenser127 ingredient342)
  (dispenses dispenser221 ingredient265)
  (clean shaker304)
  (clean shot364)
  (empty shaker304)
  (empty shot364)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker304 l0)
  (shaker-level shaker304 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail80 ingredient120)
  (cocktail-part2 cocktail80 ingredient265)
)
 (:goal
  (and
      (contains shot364 cocktail80)
)))
