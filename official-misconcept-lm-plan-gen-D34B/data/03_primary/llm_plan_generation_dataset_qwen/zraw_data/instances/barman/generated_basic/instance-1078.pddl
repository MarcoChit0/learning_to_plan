(define (problem prob)
 (:domain barman)
 (:objects 
      shaker5 - shaker
      left right - hand
      shot345 shot390 shot23 - shot
      ingredient227 ingredient46 ingredient302 ingredient171 - ingredient
      cocktail1 - cocktail
      dispenser356 dispenser86 dispenser92 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker5)
  (ontable shot345)
  (ontable shot390)
  (ontable shot23)
  (dispenses dispenser356 ingredient227)
  (dispenses dispenser86 ingredient46)
  (dispenses dispenser92 ingredient302)
  (dispenses dispenser176 ingredient171)
  (clean shaker5)
  (clean shot345)
  (clean shot390)
  (clean shot23)
  (empty shaker5)
  (empty shot345)
  (empty shot390)
  (empty shot23)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker5 l0)
  (shaker-level shaker5 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient302)
  (cocktail-part2 cocktail1 ingredient171)
)
 (:goal
  (and
      (contains shot345 cocktail1)
      (contains shot390 cocktail1)
)))
