(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot82 - shot
      ingredient249 ingredient359 - ingredient
      cocktail1 - cocktail
      dispenser337 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot82)
  (dispenses dispenser337 ingredient249)
  (dispenses dispenser323 ingredient359)
  (clean shaker172)
  (clean shot82)
  (empty shaker172)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient249)
  (cocktail-part2 cocktail1 ingredient359)
)
 (:goal
  (and
      (contains shot82 cocktail1)
)))
