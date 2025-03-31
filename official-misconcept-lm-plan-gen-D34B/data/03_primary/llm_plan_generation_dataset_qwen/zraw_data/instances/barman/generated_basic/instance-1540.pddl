(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot87 - shot
      ingredient165 ingredient395 - ingredient
      cocktail19 - cocktail
      dispenser233 dispenser237 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot87)
  (dispenses dispenser233 ingredient165)
  (dispenses dispenser237 ingredient395)
  (clean shaker294)
  (clean shot87)
  (empty shaker294)
  (empty shot87)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail19 ingredient395)
  (cocktail-part2 cocktail19 ingredient165)
)
 (:goal
  (and
      (contains shot87 cocktail19)
)))
