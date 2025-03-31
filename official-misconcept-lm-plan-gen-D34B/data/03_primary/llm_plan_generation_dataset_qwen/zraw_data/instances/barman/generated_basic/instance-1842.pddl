(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot66 shot383 - shot
      ingredient450 ingredient429 ingredient153 - ingredient
      cocktail143 - cocktail
      dispenser113 dispenser255 dispenser112 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot66)
  (ontable shot383)
  (dispenses dispenser113 ingredient450)
  (dispenses dispenser255 ingredient429)
  (dispenses dispenser112 ingredient153)
  (clean shaker433)
  (clean shot66)
  (clean shot383)
  (empty shaker433)
  (empty shot66)
  (empty shot383)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail143 ingredient153)
  (cocktail-part2 cocktail143 ingredient429)
)
 (:goal
  (and
      (contains shot66 cocktail143)
)))
