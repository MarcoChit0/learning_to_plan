(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot97 shot455 - shot
      ingredient494 ingredient9 - ingredient
      cocktail305 - cocktail
      dispenser166 dispenser448 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot97)
  (ontable shot455)
  (dispenses dispenser166 ingredient494)
  (dispenses dispenser448 ingredient9)
  (clean shaker120)
  (clean shot97)
  (clean shot455)
  (empty shaker120)
  (empty shot97)
  (empty shot455)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail305 ingredient9)
  (cocktail-part2 cocktail305 ingredient494)
)
 (:goal
  (and
      (contains shot97 cocktail305)
)))
