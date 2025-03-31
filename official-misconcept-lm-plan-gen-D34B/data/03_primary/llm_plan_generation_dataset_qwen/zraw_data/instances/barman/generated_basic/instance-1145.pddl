(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot53 shot140 - shot
      ingredient11 ingredient99 ingredient77 ingredient12 - ingredient
      cocktail1 - cocktail
      dispenser19 dispenser455 dispenser43 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot53)
  (ontable shot140)
  (dispenses dispenser19 ingredient11)
  (dispenses dispenser455 ingredient99)
  (dispenses dispenser43 ingredient77)
  (dispenses dispenser216 ingredient12)
  (clean shaker388)
  (clean shot53)
  (clean shot140)
  (empty shaker388)
  (empty shot53)
  (empty shot140)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient99)
  (cocktail-part2 cocktail1 ingredient11)
)
 (:goal
  (and
      (contains shot53 cocktail1)
)))
