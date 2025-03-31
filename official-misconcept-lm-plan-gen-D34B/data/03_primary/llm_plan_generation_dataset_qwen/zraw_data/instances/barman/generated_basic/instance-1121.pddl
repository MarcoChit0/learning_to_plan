(define (problem prob)
 (:domain barman)
 (:objects 
      shaker296 - shaker
      left right - hand
      shot72 shot248 - shot
      ingredient136 ingredient301 - ingredient
      cocktail1 - cocktail
      dispenser186 dispenser228 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker296)
  (ontable shot72)
  (ontable shot248)
  (dispenses dispenser186 ingredient136)
  (dispenses dispenser228 ingredient301)
  (clean shaker296)
  (clean shot72)
  (clean shot248)
  (empty shaker296)
  (empty shot72)
  (empty shot248)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker296 l0)
  (shaker-level shaker296 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient136)
  (cocktail-part2 cocktail1 ingredient301)
)
 (:goal
  (and
      (contains shot72 cocktail1)
)))
