(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot374 shot363 - shot
      ingredient315 ingredient404 ingredient338 - ingredient
      cocktail1 - cocktail
      dispenser256 dispenser257 dispenser376 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot374)
  (ontable shot363)
  (dispenses dispenser256 ingredient315)
  (dispenses dispenser257 ingredient404)
  (dispenses dispenser376 ingredient338)
  (clean shaker303)
  (clean shot374)
  (clean shot363)
  (empty shaker303)
  (empty shot374)
  (empty shot363)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient315)
  (cocktail-part2 cocktail1 ingredient404)
)
 (:goal
  (and
      (contains shot374 cocktail1)
)))
