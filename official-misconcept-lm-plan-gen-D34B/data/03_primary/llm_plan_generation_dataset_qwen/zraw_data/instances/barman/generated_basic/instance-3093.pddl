(define (problem prob)
 (:domain barman)
 (:objects 
      shaker366 - shaker
      left right - hand
      shot402 - shot
      ingredient495 ingredient203 ingredient277 ingredient204 - ingredient
      cocktail196 - cocktail
      dispenser424 dispenser411 dispenser21 dispenser284 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker366)
  (ontable shot402)
  (dispenses dispenser424 ingredient495)
  (dispenses dispenser411 ingredient203)
  (dispenses dispenser21 ingredient277)
  (dispenses dispenser284 ingredient204)
  (clean shaker366)
  (clean shot402)
  (empty shaker366)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker366 l0)
  (shaker-level shaker366 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail196 ingredient495)
  (cocktail-part2 cocktail196 ingredient277)
)
 (:goal
  (and
      (contains shot402 cocktail196)
)))
