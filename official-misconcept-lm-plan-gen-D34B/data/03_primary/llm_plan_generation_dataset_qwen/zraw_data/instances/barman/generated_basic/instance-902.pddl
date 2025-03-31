(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot430 shot279 - shot
      ingredient54 ingredient117 - ingredient
      cocktail1 - cocktail
      dispenser101 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot430)
  (ontable shot279)
  (dispenses dispenser101 ingredient54)
  (dispenses dispenser478 ingredient117)
  (clean shaker449)
  (clean shot430)
  (clean shot279)
  (empty shaker449)
  (empty shot430)
  (empty shot279)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient54)
  (cocktail-part2 cocktail1 ingredient117)
)
 (:goal
  (and
      (contains shot430 cocktail1)
)))
