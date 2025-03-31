(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot38 shot380 - shot
      ingredient476 ingredient496 - ingredient
      cocktail1 - cocktail
      dispenser111 dispenser462 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot38)
  (ontable shot380)
  (dispenses dispenser111 ingredient476)
  (dispenses dispenser462 ingredient496)
  (clean shaker294)
  (clean shot38)
  (clean shot380)
  (empty shaker294)
  (empty shot38)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient496)
  (cocktail-part2 cocktail1 ingredient476)
)
 (:goal
  (and
      (contains shot38 cocktail1)
)))
