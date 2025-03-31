(define (problem prob)
 (:domain barman)
 (:objects 
      shaker50 - shaker
      left right - hand
      shot108 - shot
      ingredient236 ingredient255 ingredient497 - ingredient
      cocktail1 - cocktail
      dispenser169 dispenser407 dispenser279 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker50)
  (ontable shot108)
  (dispenses dispenser169 ingredient236)
  (dispenses dispenser407 ingredient255)
  (dispenses dispenser279 ingredient497)
  (clean shaker50)
  (clean shot108)
  (empty shaker50)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker50 l0)
  (shaker-level shaker50 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient236)
  (cocktail-part2 cocktail1 ingredient497)
)
 (:goal
  (and
      (contains shot108 cocktail1)
)))
