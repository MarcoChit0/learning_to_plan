(define (problem prob)
 (:domain barman)
 (:objects 
      shaker243 - shaker
      left right - hand
      shot482 shot114 - shot
      ingredient150 ingredient453 ingredient160 - ingredient
      cocktail53 - cocktail
      dispenser349 dispenser243 dispenser457 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker243)
  (ontable shot482)
  (ontable shot114)
  (dispenses dispenser349 ingredient150)
  (dispenses dispenser243 ingredient453)
  (dispenses dispenser457 ingredient160)
  (clean shaker243)
  (clean shot482)
  (clean shot114)
  (empty shaker243)
  (empty shot482)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker243 l0)
  (shaker-level shaker243 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail53 ingredient160)
  (cocktail-part2 cocktail53 ingredient453)
)
 (:goal
  (and
      (contains shot482 cocktail53)
)))
