(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot259 shot403 - shot
      ingredient430 ingredient374 - ingredient
      cocktail1 - cocktail
      dispenser256 dispenser66 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot259)
  (ontable shot403)
  (dispenses dispenser256 ingredient430)
  (dispenses dispenser66 ingredient374)
  (clean shaker278)
  (clean shot259)
  (clean shot403)
  (empty shaker278)
  (empty shot259)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient430)
  (cocktail-part2 cocktail1 ingredient374)
)
 (:goal
  (and
      (contains shot259 cocktail1)
)))
