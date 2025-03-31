(define (problem prob)
 (:domain barman)
 (:objects 
      shaker74 - shaker
      left right - hand
      shot270 shot213 - shot
      ingredient402 ingredient73 ingredient68 ingredient284 - ingredient
      cocktail15 - cocktail
      dispenser274 dispenser19 dispenser324 dispenser96 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker74)
  (ontable shot270)
  (ontable shot213)
  (dispenses dispenser274 ingredient402)
  (dispenses dispenser19 ingredient73)
  (dispenses dispenser324 ingredient68)
  (dispenses dispenser96 ingredient284)
  (clean shaker74)
  (clean shot270)
  (clean shot213)
  (empty shaker74)
  (empty shot270)
  (empty shot213)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker74 l0)
  (shaker-level shaker74 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail15 ingredient73)
  (cocktail-part2 cocktail15 ingredient402)
)
 (:goal
  (and
      (contains shot270 cocktail15)
)))
