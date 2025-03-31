(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot371 shot412 shot269 - shot
      ingredient31 ingredient130 ingredient32 - ingredient
      cocktail1 - cocktail
      dispenser480 dispenser86 dispenser78 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot371)
  (ontable shot412)
  (ontable shot269)
  (dispenses dispenser480 ingredient31)
  (dispenses dispenser86 ingredient130)
  (dispenses dispenser78 ingredient32)
  (clean shaker375)
  (clean shot371)
  (clean shot412)
  (clean shot269)
  (empty shaker375)
  (empty shot371)
  (empty shot412)
  (empty shot269)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient32)
  (cocktail-part2 cocktail1 ingredient31)
)
 (:goal
  (and
      (contains shot371 cocktail1)
      (contains shot412 cocktail1)
)))
