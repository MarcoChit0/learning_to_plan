(define (problem prob)
 (:domain barman)
 (:objects 
      shaker17 - shaker
      left right - hand
      shot371 shot231 shot232 - shot
      ingredient201 ingredient53 ingredient12 - ingredient
      cocktail1 - cocktail
      dispenser492 dispenser287 dispenser95 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker17)
  (ontable shot371)
  (ontable shot231)
  (ontable shot232)
  (dispenses dispenser492 ingredient201)
  (dispenses dispenser287 ingredient53)
  (dispenses dispenser95 ingredient12)
  (clean shaker17)
  (clean shot371)
  (clean shot231)
  (clean shot232)
  (empty shaker17)
  (empty shot371)
  (empty shot231)
  (empty shot232)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker17 l0)
  (shaker-level shaker17 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient12)
  (cocktail-part2 cocktail1 ingredient201)
)
 (:goal
  (and
      (contains shot371 cocktail1)
      (contains shot231 cocktail1)
)))
