(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot291 shot382 - shot
      ingredient178 ingredient303 - ingredient
      cocktail67 - cocktail
      dispenser356 dispenser1 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot291)
  (ontable shot382)
  (dispenses dispenser356 ingredient178)
  (dispenses dispenser1 ingredient303)
  (clean shaker498)
  (clean shot291)
  (clean shot382)
  (empty shaker498)
  (empty shot291)
  (empty shot382)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail67 ingredient178)
  (cocktail-part2 cocktail67 ingredient303)
)
 (:goal
  (and
      (contains shot291 cocktail67)
)))
