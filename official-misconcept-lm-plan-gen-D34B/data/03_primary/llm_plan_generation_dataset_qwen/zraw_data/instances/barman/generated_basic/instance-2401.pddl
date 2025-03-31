(define (problem prob)
 (:domain barman)
 (:objects 
      shaker300 - shaker
      left right - hand
      shot409 shot167 - shot
      ingredient30 ingredient179 ingredient241 - ingredient
      cocktail465 - cocktail
      dispenser248 dispenser321 dispenser318 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker300)
  (ontable shot409)
  (ontable shot167)
  (dispenses dispenser248 ingredient30)
  (dispenses dispenser321 ingredient179)
  (dispenses dispenser318 ingredient241)
  (clean shaker300)
  (clean shot409)
  (clean shot167)
  (empty shaker300)
  (empty shot409)
  (empty shot167)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker300 l0)
  (shaker-level shaker300 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail465 ingredient179)
  (cocktail-part2 cocktail465 ingredient30)
)
 (:goal
  (and
      (contains shot409 cocktail465)
)))
