(define (problem prob)
 (:domain barman)
 (:objects 
      shaker414 - shaker
      left right - hand
      shot291 - shot
      ingredient180 ingredient187 ingredient381 - ingredient
      cocktail1 - cocktail
      dispenser173 dispenser401 dispenser238 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker414)
  (ontable shot291)
  (dispenses dispenser173 ingredient180)
  (dispenses dispenser401 ingredient187)
  (dispenses dispenser238 ingredient381)
  (clean shaker414)
  (clean shot291)
  (empty shaker414)
  (empty shot291)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker414 l0)
  (shaker-level shaker414 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient381)
  (cocktail-part2 cocktail1 ingredient187)
)
 (:goal
  (and
      (contains shot291 cocktail1)
)))
