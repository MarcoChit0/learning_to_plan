(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot213 - shot
      ingredient228 ingredient211 ingredient497 - ingredient
      cocktail88 - cocktail
      dispenser404 dispenser232 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot213)
  (dispenses dispenser404 ingredient228)
  (dispenses dispenser232 ingredient211)
  (dispenses dispenser43 ingredient497)
  (clean shaker464)
  (clean shot213)
  (empty shaker464)
  (empty shot213)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail88 ingredient228)
  (cocktail-part2 cocktail88 ingredient211)
)
 (:goal
  (and
      (contains shot213 cocktail88)
)))
