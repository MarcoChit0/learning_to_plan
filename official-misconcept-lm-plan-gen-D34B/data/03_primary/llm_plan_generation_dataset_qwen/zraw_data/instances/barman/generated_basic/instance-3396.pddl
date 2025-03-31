(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot298 shot130 - shot
      ingredient23 ingredient397 ingredient31 ingredient187 - ingredient
      cocktail363 - cocktail
      dispenser497 dispenser284 dispenser90 dispenser376 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot298)
  (ontable shot130)
  (dispenses dispenser497 ingredient23)
  (dispenses dispenser284 ingredient397)
  (dispenses dispenser90 ingredient31)
  (dispenses dispenser376 ingredient187)
  (clean shaker459)
  (clean shot298)
  (clean shot130)
  (empty shaker459)
  (empty shot298)
  (empty shot130)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail363 ingredient187)
  (cocktail-part2 cocktail363 ingredient23)
)
 (:goal
  (and
      (contains shot298 cocktail363)
)))
