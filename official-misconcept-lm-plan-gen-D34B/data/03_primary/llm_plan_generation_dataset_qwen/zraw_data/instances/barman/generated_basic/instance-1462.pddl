(define (problem prob)
 (:domain barman)
 (:objects 
      shaker478 - shaker
      left right - hand
      shot249 shot13 shot254 - shot
      ingredient247 ingredient146 - ingredient
      cocktail278 - cocktail
      dispenser74 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker478)
  (ontable shot249)
  (ontable shot13)
  (ontable shot254)
  (dispenses dispenser74 ingredient247)
  (dispenses dispenser131 ingredient146)
  (clean shaker478)
  (clean shot249)
  (clean shot13)
  (clean shot254)
  (empty shaker478)
  (empty shot249)
  (empty shot13)
  (empty shot254)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker478 l0)
  (shaker-level shaker478 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail278 ingredient247)
  (cocktail-part2 cocktail278 ingredient146)
)
 (:goal
  (and
      (contains shot249 cocktail278)
      (contains shot13 ingredient146)
)))
