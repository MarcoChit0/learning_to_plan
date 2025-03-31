(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot457 shot453 - shot
      ingredient57 ingredient367 ingredient449 - ingredient
      cocktail1 - cocktail
      dispenser360 dispenser494 dispenser256 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot457)
  (ontable shot453)
  (dispenses dispenser360 ingredient57)
  (dispenses dispenser494 ingredient367)
  (dispenses dispenser256 ingredient449)
  (clean shaker315)
  (clean shot457)
  (clean shot453)
  (empty shaker315)
  (empty shot457)
  (empty shot453)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient449)
  (cocktail-part2 cocktail1 ingredient367)
)
 (:goal
  (and
      (contains shot457 cocktail1)
)))
