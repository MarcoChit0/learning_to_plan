(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot402 - shot
      ingredient458 ingredient117 ingredient392 ingredient450 - ingredient
      cocktail1 - cocktail
      dispenser237 dispenser356 dispenser178 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot402)
  (dispenses dispenser237 ingredient458)
  (dispenses dispenser356 ingredient117)
  (dispenses dispenser178 ingredient392)
  (dispenses dispenser207 ingredient450)
  (clean shaker297)
  (clean shot402)
  (empty shaker297)
  (empty shot402)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient392)
  (cocktail-part2 cocktail1 ingredient117)
)
 (:goal
  (and
      (contains shot402 cocktail1)
)))
