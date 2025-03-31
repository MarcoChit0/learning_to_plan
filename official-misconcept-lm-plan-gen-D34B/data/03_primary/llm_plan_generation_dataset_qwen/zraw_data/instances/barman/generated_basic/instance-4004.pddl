(define (problem prob)
 (:domain barman)
 (:objects 
      shaker123 - shaker
      left right - hand
      shot1 shot136 shot290 - shot
      ingredient307 ingredient315 - ingredient
      cocktail321 - cocktail
      dispenser333 dispenser145 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker123)
  (ontable shot1)
  (ontable shot136)
  (ontable shot290)
  (dispenses dispenser333 ingredient307)
  (dispenses dispenser145 ingredient315)
  (clean shaker123)
  (clean shot1)
  (clean shot136)
  (clean shot290)
  (empty shaker123)
  (empty shot1)
  (empty shot136)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker123 l0)
  (shaker-level shaker123 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail321 ingredient307)
  (cocktail-part2 cocktail321 ingredient315)
)
 (:goal
  (and
      (contains shot1 cocktail321)
      (contains shot136 cocktail321)
)))
