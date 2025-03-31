(define (problem prob)
 (:domain barman)
 (:objects 
      shaker471 - shaker
      left right - hand
      shot389 shot491 shot424 - shot
      ingredient286 ingredient297 ingredient370 ingredient474 - ingredient
      cocktail1 - cocktail
      dispenser395 dispenser402 dispenser155 dispenser29 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker471)
  (ontable shot389)
  (ontable shot491)
  (ontable shot424)
  (dispenses dispenser395 ingredient286)
  (dispenses dispenser402 ingredient297)
  (dispenses dispenser155 ingredient370)
  (dispenses dispenser29 ingredient474)
  (clean shaker471)
  (clean shot389)
  (clean shot491)
  (clean shot424)
  (empty shaker471)
  (empty shot389)
  (empty shot491)
  (empty shot424)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker471 l0)
  (shaker-level shaker471 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient297)
  (cocktail-part2 cocktail1 ingredient286)
)
 (:goal
  (and
      (contains shot389 cocktail1)
      (contains shot491 ingredient297)
)))
