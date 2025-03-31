(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot246 shot59 shot226 - shot
      ingredient62 ingredient215 ingredient346 ingredient368 - ingredient
      cocktail1 - cocktail
      dispenser398 dispenser26 dispenser24 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot246)
  (ontable shot59)
  (ontable shot226)
  (dispenses dispenser398 ingredient62)
  (dispenses dispenser26 ingredient215)
  (dispenses dispenser24 ingredient346)
  (dispenses dispenser9 ingredient368)
  (clean shaker22)
  (clean shot246)
  (clean shot59)
  (clean shot226)
  (empty shaker22)
  (empty shot246)
  (empty shot59)
  (empty shot226)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient62)
  (cocktail-part2 cocktail1 ingredient368)
)
 (:goal
  (and
      (contains shot246 cocktail1)
      (contains shot59 cocktail1)
)))
