(define (problem prob)
 (:domain barman)
 (:objects 
      shaker222 - shaker
      left right - hand
      shot26 shot56 - shot
      ingredient103 ingredient95 ingredient404 - ingredient
      cocktail1 - cocktail
      dispenser157 dispenser383 dispenser327 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker222)
  (ontable shot26)
  (ontable shot56)
  (dispenses dispenser157 ingredient103)
  (dispenses dispenser383 ingredient95)
  (dispenses dispenser327 ingredient404)
  (clean shaker222)
  (clean shot26)
  (clean shot56)
  (empty shaker222)
  (empty shot26)
  (empty shot56)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker222 l0)
  (shaker-level shaker222 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient103)
  (cocktail-part2 cocktail1 ingredient404)
)
 (:goal
  (and
      (contains shot26 cocktail1)
)))
