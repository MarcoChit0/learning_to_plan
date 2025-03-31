(define (problem prob)
 (:domain barman)
 (:objects 
      shaker249 - shaker
      left right - hand
      shot181 - shot
      ingredient433 ingredient159 - ingredient
      cocktail46 - cocktail
      dispenser461 dispenser155 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker249)
  (ontable shot181)
  (dispenses dispenser461 ingredient433)
  (dispenses dispenser155 ingredient159)
  (clean shaker249)
  (clean shot181)
  (empty shaker249)
  (empty shot181)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker249 l0)
  (shaker-level shaker249 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail46 ingredient433)
  (cocktail-part2 cocktail46 ingredient159)
)
 (:goal
  (and
      (contains shot181 cocktail46)
)))
