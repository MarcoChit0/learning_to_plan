(define (problem prob)
 (:domain barman)
 (:objects 
      shaker3 - shaker
      left right - hand
      shot212 shot308 - shot
      ingredient485 ingredient396 ingredient230 ingredient325 - ingredient
      cocktail209 - cocktail
      dispenser89 dispenser190 dispenser479 dispenser497 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker3)
  (ontable shot212)
  (ontable shot308)
  (dispenses dispenser89 ingredient485)
  (dispenses dispenser190 ingredient396)
  (dispenses dispenser479 ingredient230)
  (dispenses dispenser497 ingredient325)
  (clean shaker3)
  (clean shot212)
  (clean shot308)
  (empty shaker3)
  (empty shot212)
  (empty shot308)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker3 l0)
  (shaker-level shaker3 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail209 ingredient325)
  (cocktail-part2 cocktail209 ingredient485)
)
 (:goal
  (and
      (contains shot212 cocktail209)
)))
