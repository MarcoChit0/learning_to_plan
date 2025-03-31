(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot92 - shot
      ingredient240 ingredient416 - ingredient
      cocktail297 - cocktail
      dispenser210 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot92)
  (dispenses dispenser210 ingredient240)
  (dispenses dispenser101 ingredient416)
  (clean shaker297)
  (clean shot92)
  (empty shaker297)
  (empty shot92)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail297 ingredient240)
  (cocktail-part2 cocktail297 ingredient416)
)
 (:goal
  (and
      (contains shot92 cocktail297)
)))
