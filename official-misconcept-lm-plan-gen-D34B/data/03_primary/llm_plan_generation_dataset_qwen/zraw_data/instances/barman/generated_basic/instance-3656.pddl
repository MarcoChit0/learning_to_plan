(define (problem prob)
 (:domain barman)
 (:objects 
      shaker256 - shaker
      left right - hand
      shot119 shot5 - shot
      ingredient120 ingredient498 ingredient452 ingredient148 - ingredient
      cocktail78 - cocktail
      dispenser206 dispenser498 dispenser443 dispenser226 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker256)
  (ontable shot119)
  (ontable shot5)
  (dispenses dispenser206 ingredient120)
  (dispenses dispenser498 ingredient498)
  (dispenses dispenser443 ingredient452)
  (dispenses dispenser226 ingredient148)
  (clean shaker256)
  (clean shot119)
  (clean shot5)
  (empty shaker256)
  (empty shot119)
  (empty shot5)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker256 l0)
  (shaker-level shaker256 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail78 ingredient120)
  (cocktail-part2 cocktail78 ingredient148)
)
 (:goal
  (and
      (contains shot119 cocktail78)
)))
