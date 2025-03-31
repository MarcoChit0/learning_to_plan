(define (problem prob)
 (:domain barman)
 (:objects 
      shaker14 - shaker
      left right - hand
      shot307 shot358 - shot
      ingredient138 ingredient188 ingredient342 - ingredient
      cocktail56 - cocktail
      dispenser179 dispenser221 dispenser413 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker14)
  (ontable shot307)
  (ontable shot358)
  (dispenses dispenser179 ingredient138)
  (dispenses dispenser221 ingredient188)
  (dispenses dispenser413 ingredient342)
  (clean shaker14)
  (clean shot307)
  (clean shot358)
  (empty shaker14)
  (empty shot307)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker14 l0)
  (shaker-level shaker14 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail56 ingredient138)
  (cocktail-part2 cocktail56 ingredient342)
)
 (:goal
  (and
      (contains shot307 cocktail56)
)))
