(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot87 shot254 - shot
      ingredient324 ingredient362 - ingredient
      cocktail199 - cocktail
      dispenser159 dispenser169 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot87)
  (ontable shot254)
  (dispenses dispenser159 ingredient324)
  (dispenses dispenser169 ingredient362)
  (clean shaker38)
  (clean shot87)
  (clean shot254)
  (empty shaker38)
  (empty shot87)
  (empty shot254)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail199 ingredient362)
  (cocktail-part2 cocktail199 ingredient324)
)
 (:goal
  (and
      (contains shot87 cocktail199)
)))
