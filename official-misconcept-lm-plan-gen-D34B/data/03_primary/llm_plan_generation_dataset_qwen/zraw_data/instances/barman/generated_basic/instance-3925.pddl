(define (problem prob)
 (:domain barman)
 (:objects 
      shaker104 - shaker
      left right - hand
      shot387 - shot
      ingredient494 ingredient14 ingredient236 ingredient104 - ingredient
      cocktail454 - cocktail
      dispenser35 dispenser76 dispenser282 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker104)
  (ontable shot387)
  (dispenses dispenser35 ingredient494)
  (dispenses dispenser76 ingredient14)
  (dispenses dispenser282 ingredient236)
  (dispenses dispenser417 ingredient104)
  (clean shaker104)
  (clean shot387)
  (empty shaker104)
  (empty shot387)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker104 l0)
  (shaker-level shaker104 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail454 ingredient236)
  (cocktail-part2 cocktail454 ingredient494)
)
 (:goal
  (and
      (contains shot387 cocktail454)
)))
