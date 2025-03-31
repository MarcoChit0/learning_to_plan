(define (problem prob)
 (:domain barman)
 (:objects 
      shaker466 - shaker
      left right - hand
      shot245 shot222 shot450 - shot
      ingredient370 ingredient469 - ingredient
      cocktail153 - cocktail
      dispenser151 dispenser95 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker466)
  (ontable shot245)
  (ontable shot222)
  (ontable shot450)
  (dispenses dispenser151 ingredient370)
  (dispenses dispenser95 ingredient469)
  (clean shaker466)
  (clean shot245)
  (clean shot222)
  (clean shot450)
  (empty shaker466)
  (empty shot245)
  (empty shot222)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker466 l0)
  (shaker-level shaker466 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail153 ingredient370)
  (cocktail-part2 cocktail153 ingredient469)
)
 (:goal
  (and
      (contains shot245 cocktail153)
      (contains shot222 cocktail153)
)))
