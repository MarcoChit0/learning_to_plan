(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot105 shot207 - shot
      ingredient312 ingredient179 - ingredient
      cocktail1 - cocktail
      dispenser239 dispenser253 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot105)
  (ontable shot207)
  (dispenses dispenser239 ingredient312)
  (dispenses dispenser253 ingredient179)
  (clean shaker329)
  (clean shot105)
  (clean shot207)
  (empty shaker329)
  (empty shot105)
  (empty shot207)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient179)
  (cocktail-part2 cocktail1 ingredient312)
)
 (:goal
  (and
      (contains shot105 cocktail1)
)))
