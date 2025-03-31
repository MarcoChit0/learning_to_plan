(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot350 shot187 - shot
      ingredient177 ingredient239 ingredient344 - ingredient
      cocktail1 - cocktail
      dispenser450 dispenser435 dispenser208 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot350)
  (ontable shot187)
  (dispenses dispenser450 ingredient177)
  (dispenses dispenser435 ingredient239)
  (dispenses dispenser208 ingredient344)
  (clean shaker162)
  (clean shot350)
  (clean shot187)
  (empty shaker162)
  (empty shot350)
  (empty shot187)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient239)
  (cocktail-part2 cocktail1 ingredient177)
)
 (:goal
  (and
      (contains shot350 cocktail1)
)))
