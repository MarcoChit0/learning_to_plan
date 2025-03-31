(define (problem prob)
 (:domain barman)
 (:objects 
      shaker197 - shaker
      left right - hand
      shot254 shot473 shot274 - shot
      ingredient306 ingredient324 - ingredient
      cocktail389 - cocktail
      dispenser49 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker197)
  (ontable shot254)
  (ontable shot473)
  (ontable shot274)
  (dispenses dispenser49 ingredient306)
  (dispenses dispenser417 ingredient324)
  (clean shaker197)
  (clean shot254)
  (clean shot473)
  (clean shot274)
  (empty shaker197)
  (empty shot254)
  (empty shot473)
  (empty shot274)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker197 l0)
  (shaker-level shaker197 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail389 ingredient306)
  (cocktail-part2 cocktail389 ingredient324)
)
 (:goal
  (and
      (contains shot254 cocktail389)
      (contains shot473 cocktail389)
)))
