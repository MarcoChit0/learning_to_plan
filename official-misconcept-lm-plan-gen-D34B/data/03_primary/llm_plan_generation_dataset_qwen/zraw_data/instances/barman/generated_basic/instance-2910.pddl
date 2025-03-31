(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot499 - shot
      ingredient360 ingredient341 - ingredient
      cocktail191 - cocktail
      dispenser80 dispenser69 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot499)
  (dispenses dispenser80 ingredient360)
  (dispenses dispenser69 ingredient341)
  (clean shaker439)
  (clean shot499)
  (empty shaker439)
  (empty shot499)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail191 ingredient360)
  (cocktail-part2 cocktail191 ingredient341)
)
 (:goal
  (and
      (contains shot499 cocktail191)
)))
