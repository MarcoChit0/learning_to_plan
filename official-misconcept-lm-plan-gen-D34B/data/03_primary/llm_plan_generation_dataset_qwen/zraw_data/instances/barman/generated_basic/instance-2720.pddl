(define (problem prob)
 (:domain barman)
 (:objects 
      shaker457 - shaker
      left right - hand
      shot196 shot222 shot71 - shot
      ingredient411 ingredient26 ingredient443 - ingredient
      cocktail216 - cocktail
      dispenser497 dispenser342 dispenser163 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker457)
  (ontable shot196)
  (ontable shot222)
  (ontable shot71)
  (dispenses dispenser497 ingredient411)
  (dispenses dispenser342 ingredient26)
  (dispenses dispenser163 ingredient443)
  (clean shaker457)
  (clean shot196)
  (clean shot222)
  (clean shot71)
  (empty shaker457)
  (empty shot196)
  (empty shot222)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker457 l0)
  (shaker-level shaker457 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail216 ingredient26)
  (cocktail-part2 cocktail216 ingredient443)
)
 (:goal
  (and
      (contains shot196 cocktail216)
      (contains shot222 cocktail216)
)))
