(define (problem prob)
 (:domain barman)
 (:objects 
      shaker282 - shaker
      left right - hand
      shot41 - shot
      ingredient99 ingredient49 ingredient424 ingredient495 - ingredient
      cocktail186 - cocktail
      dispenser401 dispenser400 dispenser373 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker282)
  (ontable shot41)
  (dispenses dispenser401 ingredient99)
  (dispenses dispenser400 ingredient49)
  (dispenses dispenser373 ingredient424)
  (dispenses dispenser385 ingredient495)
  (clean shaker282)
  (clean shot41)
  (empty shaker282)
  (empty shot41)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker282 l0)
  (shaker-level shaker282 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail186 ingredient424)
  (cocktail-part2 cocktail186 ingredient99)
)
 (:goal
  (and
      (contains shot41 cocktail186)
)))
