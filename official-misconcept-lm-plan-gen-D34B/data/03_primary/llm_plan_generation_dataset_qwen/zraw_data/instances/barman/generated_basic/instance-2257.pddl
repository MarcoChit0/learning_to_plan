(define (problem prob)
 (:domain barman)
 (:objects 
      shaker166 - shaker
      left right - hand
      shot256 shot285 shot3 - shot
      ingredient331 ingredient210 ingredient349 ingredient96 - ingredient
      cocktail269 - cocktail
      dispenser88 dispenser193 dispenser172 dispenser386 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker166)
  (ontable shot256)
  (ontable shot285)
  (ontable shot3)
  (dispenses dispenser88 ingredient331)
  (dispenses dispenser193 ingredient210)
  (dispenses dispenser172 ingredient349)
  (dispenses dispenser386 ingredient96)
  (clean shaker166)
  (clean shot256)
  (clean shot285)
  (clean shot3)
  (empty shaker166)
  (empty shot256)
  (empty shot285)
  (empty shot3)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker166 l0)
  (shaker-level shaker166 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail269 ingredient210)
  (cocktail-part2 cocktail269 ingredient331)
)
 (:goal
  (and
      (contains shot256 cocktail269)
      (contains shot285 cocktail269)
)))
