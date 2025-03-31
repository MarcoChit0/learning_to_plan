(define (problem prob)
 (:domain barman)
 (:objects 
      shaker261 - shaker
      left right - hand
      shot351 shot405 - shot
      ingredient14 ingredient132 - ingredient
      cocktail203 - cocktail
      dispenser171 dispenser483 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker261)
  (ontable shot351)
  (ontable shot405)
  (dispenses dispenser171 ingredient14)
  (dispenses dispenser483 ingredient132)
  (clean shaker261)
  (clean shot351)
  (clean shot405)
  (empty shaker261)
  (empty shot351)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker261 l0)
  (shaker-level shaker261 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail203 ingredient132)
  (cocktail-part2 cocktail203 ingredient14)
)
 (:goal
  (and
      (contains shot351 cocktail203)
)))
