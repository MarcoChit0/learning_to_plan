(define (problem prob)
 (:domain barman)
 (:objects 
      shaker231 - shaker
      left right - hand
      shot388 shot167 shot385 - shot
      ingredient349 ingredient476 ingredient48 - ingredient
      cocktail488 - cocktail
      dispenser142 dispenser210 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker231)
  (ontable shot388)
  (ontable shot167)
  (ontable shot385)
  (dispenses dispenser142 ingredient349)
  (dispenses dispenser210 ingredient476)
  (dispenses dispenser213 ingredient48)
  (clean shaker231)
  (clean shot388)
  (clean shot167)
  (clean shot385)
  (empty shaker231)
  (empty shot388)
  (empty shot167)
  (empty shot385)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker231 l0)
  (shaker-level shaker231 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail488 ingredient476)
  (cocktail-part2 cocktail488 ingredient349)
)
 (:goal
  (and
      (contains shot388 cocktail488)
      (contains shot167 cocktail488)
)))
