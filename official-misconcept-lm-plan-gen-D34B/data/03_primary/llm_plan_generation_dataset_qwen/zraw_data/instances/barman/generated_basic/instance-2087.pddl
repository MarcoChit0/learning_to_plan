(define (problem prob)
 (:domain barman)
 (:objects 
      shaker473 - shaker
      left right - hand
      shot444 shot339 - shot
      ingredient190 ingredient481 - ingredient
      cocktail90 - cocktail
      dispenser90 dispenser260 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker473)
  (ontable shot444)
  (ontable shot339)
  (dispenses dispenser90 ingredient190)
  (dispenses dispenser260 ingredient481)
  (clean shaker473)
  (clean shot444)
  (clean shot339)
  (empty shaker473)
  (empty shot444)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker473 l0)
  (shaker-level shaker473 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail90 ingredient190)
  (cocktail-part2 cocktail90 ingredient481)
)
 (:goal
  (and
      (contains shot444 cocktail90)
)))
