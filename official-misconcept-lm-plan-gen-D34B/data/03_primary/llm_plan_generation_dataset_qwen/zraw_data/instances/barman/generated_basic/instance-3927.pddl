(define (problem prob)
 (:domain barman)
 (:objects 
      shaker347 - shaker
      left right - hand
      shot16 - shot
      ingredient188 ingredient104 ingredient345 ingredient211 - ingredient
      cocktail334 - cocktail
      dispenser64 dispenser475 dispenser102 dispenser70 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker347)
  (ontable shot16)
  (dispenses dispenser64 ingredient188)
  (dispenses dispenser475 ingredient104)
  (dispenses dispenser102 ingredient345)
  (dispenses dispenser70 ingredient211)
  (clean shaker347)
  (clean shot16)
  (empty shaker347)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker347 l0)
  (shaker-level shaker347 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail334 ingredient211)
  (cocktail-part2 cocktail334 ingredient104)
)
 (:goal
  (and
      (contains shot16 cocktail334)
)))
