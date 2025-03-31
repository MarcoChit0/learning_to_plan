(define (problem prob)
 (:domain barman)
 (:objects 
      shaker259 - shaker
      left right - hand
      shot246 shot71 - shot
      ingredient36 ingredient65 ingredient113 ingredient294 - ingredient
      cocktail1 - cocktail
      dispenser353 dispenser54 dispenser258 dispenser425 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker259)
  (ontable shot246)
  (ontable shot71)
  (dispenses dispenser353 ingredient36)
  (dispenses dispenser54 ingredient65)
  (dispenses dispenser258 ingredient113)
  (dispenses dispenser425 ingredient294)
  (clean shaker259)
  (clean shot246)
  (clean shot71)
  (empty shaker259)
  (empty shot246)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker259 l0)
  (shaker-level shaker259 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient36)
  (cocktail-part2 cocktail1 ingredient294)
)
 (:goal
  (and
      (contains shot246 cocktail1)
)))
