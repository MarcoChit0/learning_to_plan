(define (problem prob)
 (:domain barman)
 (:objects 
      shaker227 - shaker
      left right - hand
      shot390 shot434 - shot
      ingredient305 ingredient496 - ingredient
      cocktail327 - cocktail
      dispenser440 dispenser269 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker227)
  (ontable shot390)
  (ontable shot434)
  (dispenses dispenser440 ingredient305)
  (dispenses dispenser269 ingredient496)
  (clean shaker227)
  (clean shot390)
  (clean shot434)
  (empty shaker227)
  (empty shot390)
  (empty shot434)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker227 l0)
  (shaker-level shaker227 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail327 ingredient496)
  (cocktail-part2 cocktail327 ingredient305)
)
 (:goal
  (and
      (contains shot390 cocktail327)
)))
