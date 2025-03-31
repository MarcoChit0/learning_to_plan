(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot418 shot447 - shot
      ingredient302 ingredient385 - ingredient
      cocktail319 - cocktail
      dispenser31 dispenser415 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot418)
  (ontable shot447)
  (dispenses dispenser31 ingredient302)
  (dispenses dispenser415 ingredient385)
  (clean shaker314)
  (clean shot418)
  (clean shot447)
  (empty shaker314)
  (empty shot418)
  (empty shot447)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail319 ingredient385)
  (cocktail-part2 cocktail319 ingredient302)
)
 (:goal
  (and
      (contains shot418 cocktail319)
)))
