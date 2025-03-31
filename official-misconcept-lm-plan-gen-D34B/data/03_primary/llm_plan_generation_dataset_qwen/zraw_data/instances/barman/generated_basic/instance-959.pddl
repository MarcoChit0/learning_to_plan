(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot86 - shot
      ingredient162 ingredient220 ingredient10 ingredient78 - ingredient
      cocktail1 - cocktail
      dispenser315 dispenser461 dispenser33 dispenser267 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot86)
  (dispenses dispenser315 ingredient162)
  (dispenses dispenser461 ingredient220)
  (dispenses dispenser33 ingredient10)
  (dispenses dispenser267 ingredient78)
  (clean shaker358)
  (clean shot86)
  (empty shaker358)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient162)
  (cocktail-part2 cocktail1 ingredient78)
)
 (:goal
  (and
      (contains shot86 cocktail1)
)))
