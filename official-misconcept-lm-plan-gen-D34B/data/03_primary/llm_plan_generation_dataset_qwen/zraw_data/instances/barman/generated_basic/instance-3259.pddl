(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot487 shot357 shot409 - shot
      ingredient390 ingredient194 ingredient95 ingredient121 - ingredient
      cocktail92 - cocktail
      dispenser424 dispenser256 dispenser165 dispenser427 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot487)
  (ontable shot357)
  (ontable shot409)
  (dispenses dispenser424 ingredient390)
  (dispenses dispenser256 ingredient194)
  (dispenses dispenser165 ingredient95)
  (dispenses dispenser427 ingredient121)
  (clean shaker113)
  (clean shot487)
  (clean shot357)
  (clean shot409)
  (empty shaker113)
  (empty shot487)
  (empty shot357)
  (empty shot409)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail92 ingredient390)
  (cocktail-part2 cocktail92 ingredient95)
)
 (:goal
  (and
      (contains shot487 cocktail92)
      (contains shot357 cocktail92)
)))
