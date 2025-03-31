(define (problem prob)
 (:domain barman)
 (:objects 
      shaker201 - shaker
      left right - hand
      shot410 shot339 - shot
      ingredient128 ingredient100 ingredient91 ingredient282 - ingredient
      cocktail459 - cocktail
      dispenser468 dispenser230 dispenser400 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker201)
  (ontable shot410)
  (ontable shot339)
  (dispenses dispenser468 ingredient128)
  (dispenses dispenser230 ingredient100)
  (dispenses dispenser400 ingredient91)
  (dispenses dispenser21 ingredient282)
  (clean shaker201)
  (clean shot410)
  (clean shot339)
  (empty shaker201)
  (empty shot410)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker201 l0)
  (shaker-level shaker201 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail459 ingredient91)
  (cocktail-part2 cocktail459 ingredient282)
)
 (:goal
  (and
      (contains shot410 cocktail459)
)))
