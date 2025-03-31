(define (problem prob)
 (:domain barman)
 (:objects 
      shaker188 - shaker
      left right - hand
      shot248 shot148 shot457 - shot
      ingredient7 ingredient33 ingredient271 - ingredient
      cocktail159 - cocktail
      dispenser152 dispenser382 dispenser482 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker188)
  (ontable shot248)
  (ontable shot148)
  (ontable shot457)
  (dispenses dispenser152 ingredient7)
  (dispenses dispenser382 ingredient33)
  (dispenses dispenser482 ingredient271)
  (clean shaker188)
  (clean shot248)
  (clean shot148)
  (clean shot457)
  (empty shaker188)
  (empty shot248)
  (empty shot148)
  (empty shot457)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker188 l0)
  (shaker-level shaker188 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail159 ingredient33)
  (cocktail-part2 cocktail159 ingredient7)
)
 (:goal
  (and
      (contains shot248 cocktail159)
      (contains shot148 cocktail159)
)))
