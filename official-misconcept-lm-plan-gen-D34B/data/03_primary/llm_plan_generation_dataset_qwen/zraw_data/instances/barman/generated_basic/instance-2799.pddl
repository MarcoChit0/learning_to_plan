(define (problem prob)
 (:domain barman)
 (:objects 
      shaker284 - shaker
      left right - hand
      shot396 shot193 - shot
      ingredient93 ingredient425 - ingredient
      cocktail415 - cocktail
      dispenser236 dispenser449 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker284)
  (ontable shot396)
  (ontable shot193)
  (dispenses dispenser236 ingredient93)
  (dispenses dispenser449 ingredient425)
  (clean shaker284)
  (clean shot396)
  (clean shot193)
  (empty shaker284)
  (empty shot396)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker284 l0)
  (shaker-level shaker284 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail415 ingredient425)
  (cocktail-part2 cocktail415 ingredient93)
)
 (:goal
  (and
      (contains shot396 cocktail415)
)))
