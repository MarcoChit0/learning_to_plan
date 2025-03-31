(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot463 - shot
      ingredient92 ingredient319 - ingredient
      cocktail262 - cocktail
      dispenser378 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot463)
  (dispenses dispenser378 ingredient92)
  (dispenses dispenser390 ingredient319)
  (clean shaker365)
  (clean shot463)
  (empty shaker365)
  (empty shot463)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail262 ingredient92)
  (cocktail-part2 cocktail262 ingredient319)
)
 (:goal
  (and
      (contains shot463 cocktail262)
)))
