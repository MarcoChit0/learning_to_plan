(define (problem prob)
 (:domain barman)
 (:objects 
      shaker364 - shaker
      left right - hand
      shot295 - shot
      ingredient120 ingredient285 ingredient82 - ingredient
      cocktail22 - cocktail
      dispenser113 dispenser100 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker364)
  (ontable shot295)
  (dispenses dispenser113 ingredient120)
  (dispenses dispenser100 ingredient285)
  (dispenses dispenser57 ingredient82)
  (clean shaker364)
  (clean shot295)
  (empty shaker364)
  (empty shot295)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker364 l0)
  (shaker-level shaker364 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail22 ingredient285)
  (cocktail-part2 cocktail22 ingredient120)
)
 (:goal
  (and
      (contains shot295 cocktail22)
)))
