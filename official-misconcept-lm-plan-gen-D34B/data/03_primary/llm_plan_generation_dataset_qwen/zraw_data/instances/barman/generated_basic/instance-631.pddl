(define (problem prob)
 (:domain barman)
 (:objects 
      shaker200 - shaker
      left right - hand
      shot436 shot390 - shot
      ingredient140 ingredient476 ingredient224 ingredient152 - ingredient
      cocktail1 - cocktail
      dispenser428 dispenser378 dispenser276 dispenser287 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker200)
  (ontable shot436)
  (ontable shot390)
  (dispenses dispenser428 ingredient140)
  (dispenses dispenser378 ingredient476)
  (dispenses dispenser276 ingredient224)
  (dispenses dispenser287 ingredient152)
  (clean shaker200)
  (clean shot436)
  (clean shot390)
  (empty shaker200)
  (empty shot436)
  (empty shot390)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker200 l0)
  (shaker-level shaker200 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient140)
  (cocktail-part2 cocktail1 ingredient152)
)
 (:goal
  (and
      (contains shot436 cocktail1)
)))
