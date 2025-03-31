(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot55 - shot
      ingredient30 ingredient66 ingredient56 ingredient451 - ingredient
      cocktail1 - cocktail
      dispenser206 dispenser403 dispenser252 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot55)
  (dispenses dispenser206 ingredient30)
  (dispenses dispenser403 ingredient66)
  (dispenses dispenser252 ingredient56)
  (dispenses dispenser214 ingredient451)
  (clean shaker22)
  (clean shot55)
  (empty shaker22)
  (empty shot55)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient66)
  (cocktail-part2 cocktail1 ingredient56)
)
 (:goal
  (and
      (contains shot55 cocktail1)
)))
