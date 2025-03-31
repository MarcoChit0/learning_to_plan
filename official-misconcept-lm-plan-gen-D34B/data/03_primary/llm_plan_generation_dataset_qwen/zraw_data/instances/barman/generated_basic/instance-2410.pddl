(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot4 - shot
      ingredient437 ingredient493 - ingredient
      cocktail354 - cocktail
      dispenser173 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot4)
  (dispenses dispenser173 ingredient437)
  (dispenses dispenser196 ingredient493)
  (clean shaker114)
  (clean shot4)
  (empty shaker114)
  (empty shot4)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail354 ingredient437)
  (cocktail-part2 cocktail354 ingredient493)
)
 (:goal
  (and
      (contains shot4 cocktail354)
)))
