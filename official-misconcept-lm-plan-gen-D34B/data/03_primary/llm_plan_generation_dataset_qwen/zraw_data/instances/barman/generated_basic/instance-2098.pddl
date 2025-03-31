(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot362 shot84 - shot
      ingredient269 ingredient207 - ingredient
      cocktail97 - cocktail
      dispenser496 dispenser320 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot362)
  (ontable shot84)
  (dispenses dispenser496 ingredient269)
  (dispenses dispenser320 ingredient207)
  (clean shaker400)
  (clean shot362)
  (clean shot84)
  (empty shaker400)
  (empty shot362)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail97 ingredient207)
  (cocktail-part2 cocktail97 ingredient269)
)
 (:goal
  (and
      (contains shot362 cocktail97)
)))
