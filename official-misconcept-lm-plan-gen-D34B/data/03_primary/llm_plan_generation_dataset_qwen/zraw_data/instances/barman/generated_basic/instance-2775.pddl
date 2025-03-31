(define (problem prob)
 (:domain barman)
 (:objects 
      shaker43 - shaker
      left right - hand
      shot388 - shot
      ingredient34 ingredient369 ingredient470 ingredient155 - ingredient
      cocktail449 - cocktail
      dispenser481 dispenser86 dispenser19 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker43)
  (ontable shot388)
  (dispenses dispenser481 ingredient34)
  (dispenses dispenser86 ingredient369)
  (dispenses dispenser19 ingredient470)
  (dispenses dispenser446 ingredient155)
  (clean shaker43)
  (clean shot388)
  (empty shaker43)
  (empty shot388)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker43 l0)
  (shaker-level shaker43 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail449 ingredient34)
  (cocktail-part2 cocktail449 ingredient369)
)
 (:goal
  (and
      (contains shot388 cocktail449)
)))
