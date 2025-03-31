(define (problem prob)
 (:domain barman)
 (:objects 
      shaker256 - shaker
      left right - hand
      shot217 shot131 - shot
      ingredient223 ingredient145 ingredient91 - ingredient
      cocktail1 - cocktail
      dispenser251 dispenser221 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker256)
  (ontable shot217)
  (ontable shot131)
  (dispenses dispenser251 ingredient223)
  (dispenses dispenser221 ingredient145)
  (dispenses dispenser130 ingredient91)
  (clean shaker256)
  (clean shot217)
  (clean shot131)
  (empty shaker256)
  (empty shot217)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker256 l0)
  (shaker-level shaker256 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient223)
  (cocktail-part2 cocktail1 ingredient91)
)
 (:goal
  (and
      (contains shot217 cocktail1)
)))
