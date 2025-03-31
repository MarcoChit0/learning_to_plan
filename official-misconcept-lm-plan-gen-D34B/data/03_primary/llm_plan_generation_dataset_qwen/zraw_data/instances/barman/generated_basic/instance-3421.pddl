(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot408 shot448 - shot
      ingredient267 ingredient145 ingredient59 - ingredient
      cocktail162 - cocktail
      dispenser29 dispenser259 dispenser90 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot408)
  (ontable shot448)
  (dispenses dispenser29 ingredient267)
  (dispenses dispenser259 ingredient145)
  (dispenses dispenser90 ingredient59)
  (clean shaker167)
  (clean shot408)
  (clean shot448)
  (empty shaker167)
  (empty shot408)
  (empty shot448)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail162 ingredient267)
  (cocktail-part2 cocktail162 ingredient145)
)
 (:goal
  (and
      (contains shot408 cocktail162)
)))
