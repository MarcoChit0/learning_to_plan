(define (problem prob)
 (:domain barman)
 (:objects 
      shaker327 - shaker
      left right - hand
      shot320 shot268 shot18 - shot
      ingredient106 ingredient385 ingredient13 - ingredient
      cocktail1 - cocktail
      dispenser82 dispenser148 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker327)
  (ontable shot320)
  (ontable shot268)
  (ontable shot18)
  (dispenses dispenser82 ingredient106)
  (dispenses dispenser148 ingredient385)
  (dispenses dispenser323 ingredient13)
  (clean shaker327)
  (clean shot320)
  (clean shot268)
  (clean shot18)
  (empty shaker327)
  (empty shot320)
  (empty shot268)
  (empty shot18)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker327 l0)
  (shaker-level shaker327 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient106)
  (cocktail-part2 cocktail1 ingredient13)
)
 (:goal
  (and
      (contains shot320 cocktail1)
      (contains shot268 cocktail1)
)))
