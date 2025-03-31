(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot288 shot101 - shot
      ingredient33 ingredient360 ingredient0 - ingredient
      cocktail18 - cocktail
      dispenser39 dispenser310 dispenser246 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot288)
  (ontable shot101)
  (dispenses dispenser39 ingredient33)
  (dispenses dispenser310 ingredient360)
  (dispenses dispenser246 ingredient0)
  (clean shaker42)
  (clean shot288)
  (clean shot101)
  (empty shaker42)
  (empty shot288)
  (empty shot101)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail18 ingredient0)
  (cocktail-part2 cocktail18 ingredient360)
)
 (:goal
  (and
      (contains shot288 cocktail18)
)))
