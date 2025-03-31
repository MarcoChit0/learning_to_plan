(define (problem prob)
 (:domain barman)
 (:objects 
      shaker178 - shaker
      left right - hand
      shot68 shot114 - shot
      ingredient490 ingredient54 - ingredient
      cocktail248 - cocktail
      dispenser179 dispenser388 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker178)
  (ontable shot68)
  (ontable shot114)
  (dispenses dispenser179 ingredient490)
  (dispenses dispenser388 ingredient54)
  (clean shaker178)
  (clean shot68)
  (clean shot114)
  (empty shaker178)
  (empty shot68)
  (empty shot114)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker178 l0)
  (shaker-level shaker178 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail248 ingredient490)
  (cocktail-part2 cocktail248 ingredient54)
)
 (:goal
  (and
      (contains shot68 cocktail248)
)))
