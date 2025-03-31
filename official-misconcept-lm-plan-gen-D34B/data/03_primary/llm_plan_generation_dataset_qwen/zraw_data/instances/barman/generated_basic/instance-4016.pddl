(define (problem prob)
 (:domain barman)
 (:objects 
      shaker124 - shaker
      left right - hand
      shot247 shot342 - shot
      ingredient96 ingredient203 ingredient265 - ingredient
      cocktail66 - cocktail
      dispenser57 dispenser491 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker124)
  (ontable shot247)
  (ontable shot342)
  (dispenses dispenser57 ingredient96)
  (dispenses dispenser491 ingredient203)
  (dispenses dispenser64 ingredient265)
  (clean shaker124)
  (clean shot247)
  (clean shot342)
  (empty shaker124)
  (empty shot247)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker124 l0)
  (shaker-level shaker124 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail66 ingredient203)
  (cocktail-part2 cocktail66 ingredient265)
)
 (:goal
  (and
      (contains shot247 cocktail66)
)))
