(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot29 shot138 - shot
      ingredient122 ingredient88 ingredient66 - ingredient
      cocktail459 - cocktail
      dispenser188 dispenser392 dispenser416 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot29)
  (ontable shot138)
  (dispenses dispenser188 ingredient122)
  (dispenses dispenser392 ingredient88)
  (dispenses dispenser416 ingredient66)
  (clean shaker400)
  (clean shot29)
  (clean shot138)
  (empty shaker400)
  (empty shot29)
  (empty shot138)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail459 ingredient88)
  (cocktail-part2 cocktail459 ingredient66)
)
 (:goal
  (and
      (contains shot29 cocktail459)
)))
