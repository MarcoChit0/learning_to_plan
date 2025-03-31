(define (problem prob)
 (:domain barman)
 (:objects 
      shaker27 - shaker
      left right - hand
      shot71 - shot
      ingredient138 ingredient236 - ingredient
      cocktail104 - cocktail
      dispenser362 dispenser316 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker27)
  (ontable shot71)
  (dispenses dispenser362 ingredient138)
  (dispenses dispenser316 ingredient236)
  (clean shaker27)
  (clean shot71)
  (empty shaker27)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker27 l0)
  (shaker-level shaker27 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail104 ingredient138)
  (cocktail-part2 cocktail104 ingredient236)
)
 (:goal
  (and
      (contains shot71 cocktail104)
)))
