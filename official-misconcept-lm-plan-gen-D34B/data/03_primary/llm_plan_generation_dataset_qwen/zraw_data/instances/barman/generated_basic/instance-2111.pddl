(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot485 shot282 shot329 - shot
      ingredient316 ingredient143 - ingredient
      cocktail134 - cocktail
      dispenser334 dispenser391 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot485)
  (ontable shot282)
  (ontable shot329)
  (dispenses dispenser334 ingredient316)
  (dispenses dispenser391 ingredient143)
  (clean shaker382)
  (clean shot485)
  (clean shot282)
  (clean shot329)
  (empty shaker382)
  (empty shot485)
  (empty shot282)
  (empty shot329)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail134 ingredient316)
  (cocktail-part2 cocktail134 ingredient143)
)
 (:goal
  (and
      (contains shot485 cocktail134)
      (contains shot282 cocktail134)
)))
