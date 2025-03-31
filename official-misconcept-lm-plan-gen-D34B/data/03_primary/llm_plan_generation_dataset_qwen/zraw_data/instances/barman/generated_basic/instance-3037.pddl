(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot151 - shot
      ingredient494 ingredient414 ingredient40 - ingredient
      cocktail251 - cocktail
      dispenser257 dispenser486 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot151)
  (dispenses dispenser257 ingredient494)
  (dispenses dispenser486 ingredient414)
  (dispenses dispenser492 ingredient40)
  (clean shaker136)
  (clean shot151)
  (empty shaker136)
  (empty shot151)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail251 ingredient414)
  (cocktail-part2 cocktail251 ingredient494)
)
 (:goal
  (and
      (contains shot151 cocktail251)
)))
