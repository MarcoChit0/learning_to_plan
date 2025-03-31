(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot64 shot397 - shot
      ingredient60 ingredient463 ingredient392 - ingredient
      cocktail1 - cocktail
      dispenser27 dispenser492 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot64)
  (ontable shot397)
  (dispenses dispenser27 ingredient60)
  (dispenses dispenser492 ingredient463)
  (dispenses dispenser216 ingredient392)
  (clean shaker332)
  (clean shot64)
  (clean shot397)
  (empty shaker332)
  (empty shot64)
  (empty shot397)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient60)
  (cocktail-part2 cocktail1 ingredient392)
)
 (:goal
  (and
      (contains shot64 cocktail1)
)))
