(define (problem prob)
 (:domain barman)
 (:objects 
      shaker85 - shaker
      left right - hand
      shot279 shot91 shot348 - shot
      ingredient151 ingredient398 - ingredient
      cocktail250 - cocktail
      dispenser480 dispenser280 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker85)
  (ontable shot279)
  (ontable shot91)
  (ontable shot348)
  (dispenses dispenser480 ingredient151)
  (dispenses dispenser280 ingredient398)
  (clean shaker85)
  (clean shot279)
  (clean shot91)
  (clean shot348)
  (empty shaker85)
  (empty shot279)
  (empty shot91)
  (empty shot348)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker85 l0)
  (shaker-level shaker85 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail250 ingredient151)
  (cocktail-part2 cocktail250 ingredient398)
)
 (:goal
  (and
      (contains shot279 cocktail250)
      (contains shot91 cocktail250)
)))
