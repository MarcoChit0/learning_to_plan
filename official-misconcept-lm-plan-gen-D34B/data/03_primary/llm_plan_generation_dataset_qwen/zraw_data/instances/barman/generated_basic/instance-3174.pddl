(define (problem prob)
 (:domain barman)
 (:objects 
      shaker381 - shaker
      left right - hand
      shot431 - shot
      ingredient18 ingredient308 ingredient206 - ingredient
      cocktail93 - cocktail
      dispenser375 dispenser335 dispenser195 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker381)
  (ontable shot431)
  (dispenses dispenser375 ingredient18)
  (dispenses dispenser335 ingredient308)
  (dispenses dispenser195 ingredient206)
  (clean shaker381)
  (clean shot431)
  (empty shaker381)
  (empty shot431)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker381 l0)
  (shaker-level shaker381 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail93 ingredient206)
  (cocktail-part2 cocktail93 ingredient308)
)
 (:goal
  (and
      (contains shot431 cocktail93)
)))
