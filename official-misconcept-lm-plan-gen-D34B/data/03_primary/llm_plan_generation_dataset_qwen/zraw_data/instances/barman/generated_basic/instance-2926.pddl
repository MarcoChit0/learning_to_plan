(define (problem prob)
 (:domain barman)
 (:objects 
      shaker279 - shaker
      left right - hand
      shot378 - shot
      ingredient196 ingredient461 - ingredient
      cocktail250 - cocktail
      dispenser139 dispenser48 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker279)
  (ontable shot378)
  (dispenses dispenser139 ingredient196)
  (dispenses dispenser48 ingredient461)
  (clean shaker279)
  (clean shot378)
  (empty shaker279)
  (empty shot378)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker279 l0)
  (shaker-level shaker279 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient461)
  (cocktail-part2 cocktail250 ingredient196)
)
 (:goal
  (and
      (contains shot378 cocktail250)
)))
