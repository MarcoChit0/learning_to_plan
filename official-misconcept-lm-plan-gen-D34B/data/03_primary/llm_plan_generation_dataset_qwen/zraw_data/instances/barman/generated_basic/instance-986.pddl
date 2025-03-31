(define (problem prob)
 (:domain barman)
 (:objects 
      shaker361 - shaker
      left right - hand
      shot22 shot398 - shot
      ingredient496 ingredient192 ingredient261 ingredient175 - ingredient
      cocktail1 - cocktail
      dispenser214 dispenser407 dispenser480 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker361)
  (ontable shot22)
  (ontable shot398)
  (dispenses dispenser214 ingredient496)
  (dispenses dispenser407 ingredient192)
  (dispenses dispenser480 ingredient261)
  (dispenses dispenser303 ingredient175)
  (clean shaker361)
  (clean shot22)
  (clean shot398)
  (empty shaker361)
  (empty shot22)
  (empty shot398)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker361 l0)
  (shaker-level shaker361 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient496)
  (cocktail-part2 cocktail1 ingredient175)
)
 (:goal
  (and
      (contains shot22 cocktail1)
)))
