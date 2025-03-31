(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot277 shot114 - shot
      ingredient97 ingredient421 ingredient27 - ingredient
      cocktail459 - cocktail
      dispenser132 dispenser120 dispenser464 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot277)
  (ontable shot114)
  (dispenses dispenser132 ingredient97)
  (dispenses dispenser120 ingredient421)
  (dispenses dispenser464 ingredient27)
  (clean shaker384)
  (clean shot277)
  (clean shot114)
  (empty shaker384)
  (empty shot277)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail459 ingredient97)
  (cocktail-part2 cocktail459 ingredient421)
)
 (:goal
  (and
      (contains shot277 cocktail459)
)))
