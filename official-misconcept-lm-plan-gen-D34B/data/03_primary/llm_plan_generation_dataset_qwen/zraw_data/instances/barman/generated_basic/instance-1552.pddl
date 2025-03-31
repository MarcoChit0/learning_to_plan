(define (problem prob)
 (:domain barman)
 (:objects 
      shaker462 - shaker
      left right - hand
      shot176 - shot
      ingredient3 ingredient301 - ingredient
      cocktail439 - cocktail
      dispenser132 dispenser112 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker462)
  (ontable shot176)
  (dispenses dispenser132 ingredient3)
  (dispenses dispenser112 ingredient301)
  (clean shaker462)
  (clean shot176)
  (empty shaker462)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker462 l0)
  (shaker-level shaker462 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail439 ingredient3)
  (cocktail-part2 cocktail439 ingredient301)
)
 (:goal
  (and
      (contains shot176 cocktail439)
)))
