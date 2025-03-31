(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot215 - shot
      ingredient127 ingredient337 - ingredient
      cocktail1 - cocktail
      dispenser33 dispenser222 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot215)
  (dispenses dispenser33 ingredient127)
  (dispenses dispenser222 ingredient337)
  (clean shaker385)
  (clean shot215)
  (empty shaker385)
  (empty shot215)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient337)
  (cocktail-part2 cocktail1 ingredient127)
)
 (:goal
  (and
      (contains shot215 cocktail1)
)))
