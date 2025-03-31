(define (problem prob)
 (:domain barman)
 (:objects 
      shaker7 - shaker
      left right - hand
      shot109 - shot
      ingredient145 ingredient140 - ingredient
      cocktail365 - cocktail
      dispenser291 dispenser282 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker7)
  (ontable shot109)
  (dispenses dispenser291 ingredient145)
  (dispenses dispenser282 ingredient140)
  (clean shaker7)
  (clean shot109)
  (empty shaker7)
  (empty shot109)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker7 l0)
  (shaker-level shaker7 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail365 ingredient145)
  (cocktail-part2 cocktail365 ingredient140)
)
 (:goal
  (and
      (contains shot109 cocktail365)
)))
