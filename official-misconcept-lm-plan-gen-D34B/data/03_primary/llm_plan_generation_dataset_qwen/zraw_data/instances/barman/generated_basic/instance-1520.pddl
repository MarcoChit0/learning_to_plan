(define (problem prob)
 (:domain barman)
 (:objects 
      shaker325 - shaker
      left right - hand
      shot181 shot216 - shot
      ingredient18 ingredient401 ingredient230 ingredient202 - ingredient
      cocktail412 - cocktail
      dispenser113 dispenser402 dispenser136 dispenser489 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker325)
  (ontable shot181)
  (ontable shot216)
  (dispenses dispenser113 ingredient18)
  (dispenses dispenser402 ingredient401)
  (dispenses dispenser136 ingredient230)
  (dispenses dispenser489 ingredient202)
  (clean shaker325)
  (clean shot181)
  (clean shot216)
  (empty shaker325)
  (empty shot181)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker325 l0)
  (shaker-level shaker325 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail412 ingredient230)
  (cocktail-part2 cocktail412 ingredient401)
)
 (:goal
  (and
      (contains shot181 cocktail412)
)))
