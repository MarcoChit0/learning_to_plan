(define (problem prob)
 (:domain barman)
 (:objects 
      shaker44 - shaker
      left right - hand
      shot311 - shot
      ingredient456 ingredient141 ingredient130 - ingredient
      cocktail1 - cocktail
      dispenser428 dispenser199 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker44)
  (ontable shot311)
  (dispenses dispenser428 ingredient456)
  (dispenses dispenser199 ingredient141)
  (dispenses dispenser217 ingredient130)
  (clean shaker44)
  (clean shot311)
  (empty shaker44)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker44 l0)
  (shaker-level shaker44 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient130)
  (cocktail-part2 cocktail1 ingredient456)
)
 (:goal
  (and
      (contains shot311 cocktail1)
)))
