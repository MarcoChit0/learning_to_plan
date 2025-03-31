(define (problem prob)
 (:domain barman)
 (:objects 
      shaker344 - shaker
      left right - hand
      shot253 shot172 shot133 - shot
      ingredient151 ingredient12 - ingredient
      cocktail441 - cocktail
      dispenser367 dispenser347 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker344)
  (ontable shot253)
  (ontable shot172)
  (ontable shot133)
  (dispenses dispenser367 ingredient151)
  (dispenses dispenser347 ingredient12)
  (clean shaker344)
  (clean shot253)
  (clean shot172)
  (clean shot133)
  (empty shaker344)
  (empty shot253)
  (empty shot172)
  (empty shot133)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker344 l0)
  (shaker-level shaker344 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail441 ingredient151)
  (cocktail-part2 cocktail441 ingredient12)
)
 (:goal
  (and
      (contains shot253 cocktail441)
      (contains shot172 cocktail441)
)))
