(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot345 shot266 - shot
      ingredient337 ingredient377 ingredient179 - ingredient
      cocktail1 - cocktail
      dispenser286 dispenser227 dispenser283 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot345)
  (ontable shot266)
  (dispenses dispenser286 ingredient337)
  (dispenses dispenser227 ingredient377)
  (dispenses dispenser283 ingredient179)
  (clean shaker305)
  (clean shot345)
  (clean shot266)
  (empty shaker305)
  (empty shot345)
  (empty shot266)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient377)
  (cocktail-part2 cocktail1 ingredient179)
)
 (:goal
  (and
      (contains shot345 cocktail1)
)))
