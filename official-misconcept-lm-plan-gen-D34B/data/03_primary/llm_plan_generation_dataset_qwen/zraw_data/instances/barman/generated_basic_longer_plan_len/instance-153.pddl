(define (problem prob)
 (:domain barman)
 (:objects 
      shaker357 - shaker
      left right - hand
      shot318 shot47 shot103 - shot
      ingredient156 ingredient488 ingredient343 ingredient374 - ingredient
      cocktail1 - cocktail
      dispenser135 dispenser333 dispenser104 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker357)
  (ontable shot318)
  (ontable shot47)
  (ontable shot103)
  (dispenses dispenser135 ingredient156)
  (dispenses dispenser333 ingredient488)
  (dispenses dispenser104 ingredient343)
  (dispenses dispenser43 ingredient374)
  (clean shaker357)
  (clean shot318)
  (clean shot47)
  (clean shot103)
  (empty shaker357)
  (empty shot318)
  (empty shot47)
  (empty shot103)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker357 l0)
  (shaker-level shaker357 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient488)
  (cocktail-part2 cocktail1 ingredient374)
)
 (:goal
  (and
      (contains shot318 cocktail1)
      (contains shot47 ingredient156)
)))
