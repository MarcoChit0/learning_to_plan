(define (problem prob)
 (:domain barman)
 (:objects 
      shaker236 - shaker
      left right - hand
      shot490 shot437 - shot
      ingredient110 ingredient480 ingredient419 ingredient279 - ingredient
      cocktail423 - cocktail
      dispenser324 dispenser192 dispenser17 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker236)
  (ontable shot490)
  (ontable shot437)
  (dispenses dispenser324 ingredient110)
  (dispenses dispenser192 ingredient480)
  (dispenses dispenser17 ingredient419)
  (dispenses dispenser422 ingredient279)
  (clean shaker236)
  (clean shot490)
  (clean shot437)
  (empty shaker236)
  (empty shot490)
  (empty shot437)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker236 l0)
  (shaker-level shaker236 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail423 ingredient279)
  (cocktail-part2 cocktail423 ingredient480)
)
 (:goal
  (and
      (contains shot490 cocktail423)
)))
