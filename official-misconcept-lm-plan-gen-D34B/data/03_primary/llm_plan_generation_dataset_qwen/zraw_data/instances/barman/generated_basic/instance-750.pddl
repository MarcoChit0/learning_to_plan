(define (problem prob)
 (:domain barman)
 (:objects 
      shaker199 - shaker
      left right - hand
      shot358 shot149 - shot
      ingredient206 ingredient85 ingredient371 - ingredient
      cocktail1 - cocktail
      dispenser448 dispenser259 dispenser345 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker199)
  (ontable shot358)
  (ontable shot149)
  (dispenses dispenser448 ingredient206)
  (dispenses dispenser259 ingredient85)
  (dispenses dispenser345 ingredient371)
  (clean shaker199)
  (clean shot358)
  (clean shot149)
  (empty shaker199)
  (empty shot358)
  (empty shot149)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker199 l0)
  (shaker-level shaker199 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient85)
  (cocktail-part2 cocktail1 ingredient206)
)
 (:goal
  (and
      (contains shot358 cocktail1)
)))
