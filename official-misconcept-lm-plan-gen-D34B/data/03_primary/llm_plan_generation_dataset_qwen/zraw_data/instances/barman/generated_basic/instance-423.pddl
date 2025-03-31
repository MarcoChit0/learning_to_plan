(define (problem prob)
 (:domain barman)
 (:objects 
      shaker239 - shaker
      left right - hand
      shot15 - shot
      ingredient347 ingredient404 - ingredient
      cocktail1 - cocktail
      dispenser177 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker239)
  (ontable shot15)
  (dispenses dispenser177 ingredient347)
  (dispenses dispenser420 ingredient404)
  (clean shaker239)
  (clean shot15)
  (empty shaker239)
  (empty shot15)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker239 l0)
  (shaker-level shaker239 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient347)
  (cocktail-part2 cocktail1 ingredient404)
)
 (:goal
  (and
      (contains shot15 cocktail1)
)))
