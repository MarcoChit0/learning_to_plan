(define (problem prob)
 (:domain barman)
 (:objects 
      shaker293 - shaker
      left right - hand
      shot441 shot58 - shot
      ingredient334 ingredient477 ingredient150 ingredient74 - ingredient
      cocktail24 - cocktail
      dispenser199 dispenser73 dispenser383 dispenser297 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker293)
  (ontable shot441)
  (ontable shot58)
  (dispenses dispenser199 ingredient334)
  (dispenses dispenser73 ingredient477)
  (dispenses dispenser383 ingredient150)
  (dispenses dispenser297 ingredient74)
  (clean shaker293)
  (clean shot441)
  (clean shot58)
  (empty shaker293)
  (empty shot441)
  (empty shot58)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker293 l0)
  (shaker-level shaker293 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail24 ingredient477)
  (cocktail-part2 cocktail24 ingredient150)
)
 (:goal
  (and
      (contains shot441 cocktail24)
)))
