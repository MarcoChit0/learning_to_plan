(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot263 shot179 - shot
      ingredient170 ingredient163 ingredient248 ingredient232 - ingredient
      cocktail75 - cocktail
      dispenser320 dispenser406 dispenser17 dispenser453 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot263)
  (ontable shot179)
  (dispenses dispenser320 ingredient170)
  (dispenses dispenser406 ingredient163)
  (dispenses dispenser17 ingredient248)
  (dispenses dispenser453 ingredient232)
  (clean shaker98)
  (clean shot263)
  (clean shot179)
  (empty shaker98)
  (empty shot263)
  (empty shot179)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail75 ingredient232)
  (cocktail-part2 cocktail75 ingredient248)
)
 (:goal
  (and
      (contains shot263 cocktail75)
)))
