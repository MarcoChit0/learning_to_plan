(define (problem prob)
 (:domain barman)
 (:objects 
      shaker198 - shaker
      left right - hand
      shot73 shot248 - shot
      ingredient323 ingredient344 - ingredient
      cocktail257 - cocktail
      dispenser195 dispenser192 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker198)
  (ontable shot73)
  (ontable shot248)
  (dispenses dispenser195 ingredient323)
  (dispenses dispenser192 ingredient344)
  (clean shaker198)
  (clean shot73)
  (clean shot248)
  (empty shaker198)
  (empty shot73)
  (empty shot248)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker198 l0)
  (shaker-level shaker198 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail257 ingredient344)
  (cocktail-part2 cocktail257 ingredient323)
)
 (:goal
  (and
      (contains shot73 cocktail257)
)))
