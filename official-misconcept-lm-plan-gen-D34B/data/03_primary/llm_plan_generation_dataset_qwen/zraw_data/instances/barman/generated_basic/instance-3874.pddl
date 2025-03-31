(define (problem prob)
 (:domain barman)
 (:objects 
      shaker201 - shaker
      left right - hand
      shot162 shot138 - shot
      ingredient218 ingredient366 ingredient152 - ingredient
      cocktail171 - cocktail
      dispenser340 dispenser390 dispenser498 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker201)
  (ontable shot162)
  (ontable shot138)
  (dispenses dispenser340 ingredient218)
  (dispenses dispenser390 ingredient366)
  (dispenses dispenser498 ingredient152)
  (clean shaker201)
  (clean shot162)
  (clean shot138)
  (empty shaker201)
  (empty shot162)
  (empty shot138)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker201 l0)
  (shaker-level shaker201 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail171 ingredient366)
  (cocktail-part2 cocktail171 ingredient218)
)
 (:goal
  (and
      (contains shot162 cocktail171)
)))
