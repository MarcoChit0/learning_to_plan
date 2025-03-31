(define (problem prob)
 (:domain barman)
 (:objects 
      shaker106 - shaker
      left right - hand
      shot429 shot124 shot94 - shot
      ingredient499 ingredient471 ingredient13 ingredient391 - ingredient
      cocktail418 - cocktail
      dispenser244 dispenser205 dispenser395 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker106)
  (ontable shot429)
  (ontable shot124)
  (ontable shot94)
  (dispenses dispenser244 ingredient499)
  (dispenses dispenser205 ingredient471)
  (dispenses dispenser395 ingredient13)
  (dispenses dispenser385 ingredient391)
  (clean shaker106)
  (clean shot429)
  (clean shot124)
  (clean shot94)
  (empty shaker106)
  (empty shot429)
  (empty shot124)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker106 l0)
  (shaker-level shaker106 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail418 ingredient471)
  (cocktail-part2 cocktail418 ingredient391)
)
 (:goal
  (and
      (contains shot429 cocktail418)
      (contains shot124 ingredient499)
)))
