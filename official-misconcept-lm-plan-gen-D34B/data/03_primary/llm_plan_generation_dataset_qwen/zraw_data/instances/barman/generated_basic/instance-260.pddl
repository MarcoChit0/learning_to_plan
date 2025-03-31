(define (problem prob)
 (:domain barman)
 (:objects 
      shaker447 - shaker
      left right - hand
      shot217 shot181 shot403 - shot
      ingredient403 ingredient339 ingredient497 - ingredient
      cocktail1 - cocktail
      dispenser199 dispenser198 dispenser411 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker447)
  (ontable shot217)
  (ontable shot181)
  (ontable shot403)
  (dispenses dispenser199 ingredient403)
  (dispenses dispenser198 ingredient339)
  (dispenses dispenser411 ingredient497)
  (clean shaker447)
  (clean shot217)
  (clean shot181)
  (clean shot403)
  (empty shaker447)
  (empty shot217)
  (empty shot181)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker447 l0)
  (shaker-level shaker447 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient497)
  (cocktail-part2 cocktail1 ingredient339)
)
 (:goal
  (and
      (contains shot217 cocktail1)
      (contains shot181 cocktail1)
)))
