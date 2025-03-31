(define (problem prob)
 (:domain barman)
 (:objects 
      shaker325 - shaker
      left right - hand
      shot440 shot391 - shot
      ingredient127 ingredient389 ingredient325 ingredient211 - ingredient
      cocktail200 - cocktail
      dispenser163 dispenser245 dispenser225 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker325)
  (ontable shot440)
  (ontable shot391)
  (dispenses dispenser163 ingredient127)
  (dispenses dispenser245 ingredient389)
  (dispenses dispenser225 ingredient325)
  (dispenses dispenser115 ingredient211)
  (clean shaker325)
  (clean shot440)
  (clean shot391)
  (empty shaker325)
  (empty shot440)
  (empty shot391)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker325 l0)
  (shaker-level shaker325 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail200 ingredient127)
  (cocktail-part2 cocktail200 ingredient325)
)
 (:goal
  (and
      (contains shot440 cocktail200)
)))
