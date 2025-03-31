(define (problem prob)
 (:domain barman)
 (:objects 
      shaker462 - shaker
      left right - hand
      shot457 - shot
      ingredient290 ingredient460 ingredient233 ingredient464 - ingredient
      cocktail64 - cocktail
      dispenser344 dispenser185 dispenser97 dispenser4 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker462)
  (ontable shot457)
  (dispenses dispenser344 ingredient290)
  (dispenses dispenser185 ingredient460)
  (dispenses dispenser97 ingredient233)
  (dispenses dispenser4 ingredient464)
  (clean shaker462)
  (clean shot457)
  (empty shaker462)
  (empty shot457)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker462 l0)
  (shaker-level shaker462 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail64 ingredient464)
  (cocktail-part2 cocktail64 ingredient233)
)
 (:goal
  (and
      (contains shot457 cocktail64)
)))
