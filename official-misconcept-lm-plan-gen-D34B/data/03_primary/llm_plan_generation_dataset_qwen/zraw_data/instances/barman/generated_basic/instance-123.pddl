(define (problem prob)
 (:domain barman)
 (:objects 
      shaker73 - shaker
      left right - hand
      shot229 shot131 - shot
      ingredient490 ingredient378 - ingredient
      cocktail1 - cocktail
      dispenser117 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker73)
  (ontable shot229)
  (ontable shot131)
  (dispenses dispenser117 ingredient490)
  (dispenses dispenser9 ingredient378)
  (clean shaker73)
  (clean shot229)
  (clean shot131)
  (empty shaker73)
  (empty shot229)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker73 l0)
  (shaker-level shaker73 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient490)
  (cocktail-part2 cocktail1 ingredient378)
)
 (:goal
  (and
      (contains shot229 cocktail1)
)))
