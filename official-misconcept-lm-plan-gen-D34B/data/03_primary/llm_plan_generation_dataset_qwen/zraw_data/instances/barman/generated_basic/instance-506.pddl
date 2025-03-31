(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot148 shot457 - shot
      ingredient319 ingredient213 ingredient18 ingredient102 - ingredient
      cocktail1 - cocktail
      dispenser192 dispenser51 dispenser216 dispenser141 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot148)
  (ontable shot457)
  (dispenses dispenser192 ingredient319)
  (dispenses dispenser51 ingredient213)
  (dispenses dispenser216 ingredient18)
  (dispenses dispenser141 ingredient102)
  (clean shaker19)
  (clean shot148)
  (clean shot457)
  (empty shaker19)
  (empty shot148)
  (empty shot457)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient213)
  (cocktail-part2 cocktail1 ingredient102)
)
 (:goal
  (and
      (contains shot148 cocktail1)
)))
