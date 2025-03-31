(define (problem prob)
 (:domain barman)
 (:objects 
      shaker327 - shaker
      left right - hand
      shot418 shot328 shot344 - shot
      ingredient236 ingredient100 - ingredient
      cocktail69 - cocktail
      dispenser354 dispenser66 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker327)
  (ontable shot418)
  (ontable shot328)
  (ontable shot344)
  (dispenses dispenser354 ingredient236)
  (dispenses dispenser66 ingredient100)
  (clean shaker327)
  (clean shot418)
  (clean shot328)
  (clean shot344)
  (empty shaker327)
  (empty shot418)
  (empty shot328)
  (empty shot344)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker327 l0)
  (shaker-level shaker327 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail69 ingredient236)
  (cocktail-part2 cocktail69 ingredient100)
)
 (:goal
  (and
      (contains shot418 cocktail69)
      (contains shot328 ingredient100)
)))
