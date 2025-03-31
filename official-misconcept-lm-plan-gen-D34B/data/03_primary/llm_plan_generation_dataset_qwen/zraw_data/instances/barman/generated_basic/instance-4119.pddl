(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot91 shot318 - shot
      ingredient499 ingredient19 ingredient146 - ingredient
      cocktail186 - cocktail
      dispenser158 dispenser168 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot91)
  (ontable shot318)
  (dispenses dispenser158 ingredient499)
  (dispenses dispenser168 ingredient19)
  (dispenses dispenser214 ingredient146)
  (clean shaker411)
  (clean shot91)
  (clean shot318)
  (empty shaker411)
  (empty shot91)
  (empty shot318)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail186 ingredient146)
  (cocktail-part2 cocktail186 ingredient499)
)
 (:goal
  (and
      (contains shot91 cocktail186)
)))
