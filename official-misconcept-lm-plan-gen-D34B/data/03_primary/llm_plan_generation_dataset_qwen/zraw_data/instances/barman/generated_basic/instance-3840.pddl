(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot95 shot454 shot116 - shot
      ingredient320 ingredient307 - ingredient
      cocktail230 - cocktail
      dispenser145 dispenser298 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot95)
  (ontable shot454)
  (ontable shot116)
  (dispenses dispenser145 ingredient320)
  (dispenses dispenser298 ingredient307)
  (clean shaker498)
  (clean shot95)
  (clean shot454)
  (clean shot116)
  (empty shaker498)
  (empty shot95)
  (empty shot454)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail230 ingredient320)
  (cocktail-part2 cocktail230 ingredient307)
)
 (:goal
  (and
      (contains shot95 cocktail230)
      (contains shot454 cocktail230)
)))
