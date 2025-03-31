(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot20 shot396 shot56 - shot
      ingredient360 ingredient298 - ingredient
      cocktail400 - cocktail
      dispenser431 dispenser129 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot20)
  (ontable shot396)
  (ontable shot56)
  (dispenses dispenser431 ingredient360)
  (dispenses dispenser129 ingredient298)
  (clean shaker384)
  (clean shot20)
  (clean shot396)
  (clean shot56)
  (empty shaker384)
  (empty shot20)
  (empty shot396)
  (empty shot56)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail400 ingredient298)
  (cocktail-part2 cocktail400 ingredient360)
)
 (:goal
  (and
      (contains shot20 cocktail400)
      (contains shot396 cocktail400)
)))
