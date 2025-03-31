(define (problem prob)
 (:domain barman)
 (:objects 
      shaker374 - shaker
      left right - hand
      shot417 shot292 shot491 - shot
      ingredient28 ingredient159 - ingredient
      cocktail110 - cocktail
      dispenser129 dispenser280 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker374)
  (ontable shot417)
  (ontable shot292)
  (ontable shot491)
  (dispenses dispenser129 ingredient28)
  (dispenses dispenser280 ingredient159)
  (clean shaker374)
  (clean shot417)
  (clean shot292)
  (clean shot491)
  (empty shaker374)
  (empty shot417)
  (empty shot292)
  (empty shot491)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker374 l0)
  (shaker-level shaker374 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail110 ingredient28)
  (cocktail-part2 cocktail110 ingredient159)
)
 (:goal
  (and
      (contains shot417 cocktail110)
      (contains shot292 cocktail110)
)))
