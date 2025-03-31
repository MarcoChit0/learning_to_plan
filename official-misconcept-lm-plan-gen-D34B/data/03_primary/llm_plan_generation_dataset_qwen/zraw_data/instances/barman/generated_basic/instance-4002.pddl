(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot100 shot88 shot109 - shot
      ingredient493 ingredient416 - ingredient
      cocktail28 - cocktail
      dispenser498 dispenser108 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot100)
  (ontable shot88)
  (ontable shot109)
  (dispenses dispenser498 ingredient493)
  (dispenses dispenser108 ingredient416)
  (clean shaker151)
  (clean shot100)
  (clean shot88)
  (clean shot109)
  (empty shaker151)
  (empty shot100)
  (empty shot88)
  (empty shot109)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail28 ingredient416)
  (cocktail-part2 cocktail28 ingredient493)
)
 (:goal
  (and
      (contains shot100 cocktail28)
      (contains shot88 cocktail28)
)))
