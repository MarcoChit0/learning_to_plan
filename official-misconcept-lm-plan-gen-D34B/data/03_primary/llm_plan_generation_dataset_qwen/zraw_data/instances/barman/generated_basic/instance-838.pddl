(define (problem prob)
 (:domain barman)
 (:objects 
      shaker34 - shaker
      left right - hand
      shot453 shot269 shot496 - shot
      ingredient284 ingredient451 ingredient456 ingredient205 - ingredient
      cocktail1 - cocktail
      dispenser276 dispenser392 dispenser123 dispenser74 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker34)
  (ontable shot453)
  (ontable shot269)
  (ontable shot496)
  (dispenses dispenser276 ingredient284)
  (dispenses dispenser392 ingredient451)
  (dispenses dispenser123 ingredient456)
  (dispenses dispenser74 ingredient205)
  (clean shaker34)
  (clean shot453)
  (clean shot269)
  (clean shot496)
  (empty shaker34)
  (empty shot453)
  (empty shot269)
  (empty shot496)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker34 l0)
  (shaker-level shaker34 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient205)
  (cocktail-part2 cocktail1 ingredient456)
)
 (:goal
  (and
      (contains shot453 cocktail1)
      (contains shot269 cocktail1)
)))
