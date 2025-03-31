(define (problem prob)
 (:domain barman)
 (:objects 
      shaker229 - shaker
      left right - hand
      shot112 - shot
      ingredient404 ingredient243 - ingredient
      cocktail1 - cocktail
      dispenser399 dispenser98 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker229)
  (ontable shot112)
  (dispenses dispenser399 ingredient404)
  (dispenses dispenser98 ingredient243)
  (clean shaker229)
  (clean shot112)
  (empty shaker229)
  (empty shot112)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker229 l0)
  (shaker-level shaker229 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient243)
  (cocktail-part2 cocktail1 ingredient404)
)
 (:goal
  (and
      (contains shot112 cocktail1)
)))
