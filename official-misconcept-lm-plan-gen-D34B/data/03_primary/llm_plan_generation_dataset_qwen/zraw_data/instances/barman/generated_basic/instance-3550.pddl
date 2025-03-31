(define (problem prob)
 (:domain barman)
 (:objects 
      shaker192 - shaker
      left right - hand
      shot49 - shot
      ingredient135 ingredient315 ingredient323 - ingredient
      cocktail288 - cocktail
      dispenser430 dispenser487 dispenser117 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker192)
  (ontable shot49)
  (dispenses dispenser430 ingredient135)
  (dispenses dispenser487 ingredient315)
  (dispenses dispenser117 ingredient323)
  (clean shaker192)
  (clean shot49)
  (empty shaker192)
  (empty shot49)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker192 l0)
  (shaker-level shaker192 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail288 ingredient135)
  (cocktail-part2 cocktail288 ingredient315)
)
 (:goal
  (and
      (contains shot49 cocktail288)
)))
