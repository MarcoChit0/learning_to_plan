(define (problem prob)
 (:domain barman)
 (:objects 
      shaker147 - shaker
      left right - hand
      shot269 shot273 shot350 - shot
      ingredient363 ingredient0 ingredient29 - ingredient
      cocktail86 - cocktail
      dispenser457 dispenser47 dispenser243 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker147)
  (ontable shot269)
  (ontable shot273)
  (ontable shot350)
  (dispenses dispenser457 ingredient363)
  (dispenses dispenser47 ingredient0)
  (dispenses dispenser243 ingredient29)
  (clean shaker147)
  (clean shot269)
  (clean shot273)
  (clean shot350)
  (empty shaker147)
  (empty shot269)
  (empty shot273)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker147 l0)
  (shaker-level shaker147 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail86 ingredient0)
  (cocktail-part2 cocktail86 ingredient29)
)
 (:goal
  (and
      (contains shot269 cocktail86)
      (contains shot273 ingredient29)
)))
