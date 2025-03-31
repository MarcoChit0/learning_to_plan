(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot278 shot328 shot349 - shot
      ingredient426 ingredient204 - ingredient
      cocktail370 - cocktail
      dispenser52 dispenser348 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot278)
  (ontable shot328)
  (ontable shot349)
  (dispenses dispenser52 ingredient426)
  (dispenses dispenser348 ingredient204)
  (clean shaker411)
  (clean shot278)
  (clean shot328)
  (clean shot349)
  (empty shaker411)
  (empty shot278)
  (empty shot328)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail370 ingredient204)
  (cocktail-part2 cocktail370 ingredient426)
)
 (:goal
  (and
      (contains shot278 cocktail370)
      (contains shot328 ingredient426)
)))
