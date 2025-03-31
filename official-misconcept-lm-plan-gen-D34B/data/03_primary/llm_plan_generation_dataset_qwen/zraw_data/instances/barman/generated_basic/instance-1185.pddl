(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot247 - shot
      ingredient42 ingredient199 ingredient470 - ingredient
      cocktail1 - cocktail
      dispenser351 dispenser191 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot247)
  (dispenses dispenser351 ingredient42)
  (dispenses dispenser191 ingredient199)
  (dispenses dispenser429 ingredient470)
  (clean shaker64)
  (clean shot247)
  (empty shaker64)
  (empty shot247)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient199)
  (cocktail-part2 cocktail1 ingredient470)
)
 (:goal
  (and
      (contains shot247 cocktail1)
)))
