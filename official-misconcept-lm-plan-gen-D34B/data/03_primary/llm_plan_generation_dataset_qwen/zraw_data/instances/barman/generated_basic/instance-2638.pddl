(define (problem prob)
 (:domain barman)
 (:objects 
      shaker412 - shaker
      left right - hand
      shot233 shot329 - shot
      ingredient220 ingredient54 ingredient10 - ingredient
      cocktail113 - cocktail
      dispenser187 dispenser204 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker412)
  (ontable shot233)
  (ontable shot329)
  (dispenses dispenser187 ingredient220)
  (dispenses dispenser204 ingredient54)
  (dispenses dispenser283 ingredient10)
  (clean shaker412)
  (clean shot233)
  (clean shot329)
  (empty shaker412)
  (empty shot233)
  (empty shot329)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker412 l0)
  (shaker-level shaker412 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail113 ingredient54)
  (cocktail-part2 cocktail113 ingredient220)
)
 (:goal
  (and
      (contains shot233 cocktail113)
)))
