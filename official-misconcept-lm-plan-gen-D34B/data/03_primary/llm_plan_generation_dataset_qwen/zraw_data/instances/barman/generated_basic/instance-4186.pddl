(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot360 - shot
      ingredient359 ingredient342 ingredient307 ingredient201 - ingredient
      cocktail493 - cocktail
      dispenser221 dispenser388 dispenser69 dispenser66 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot360)
  (dispenses dispenser221 ingredient359)
  (dispenses dispenser388 ingredient342)
  (dispenses dispenser69 ingredient307)
  (dispenses dispenser66 ingredient201)
  (clean shaker303)
  (clean shot360)
  (empty shaker303)
  (empty shot360)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail493 ingredient201)
  (cocktail-part2 cocktail493 ingredient307)
)
 (:goal
  (and
      (contains shot360 cocktail493)
)))
