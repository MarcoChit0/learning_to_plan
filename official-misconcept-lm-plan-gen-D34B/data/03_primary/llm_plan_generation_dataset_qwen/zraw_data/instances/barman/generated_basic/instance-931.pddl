(define (problem prob)
 (:domain barman)
 (:objects 
      shaker57 - shaker
      left right - hand
      shot10 shot156 shot101 - shot
      ingredient153 ingredient498 ingredient3 ingredient106 - ingredient
      cocktail1 - cocktail
      dispenser368 dispenser430 dispenser389 dispenser310 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker57)
  (ontable shot10)
  (ontable shot156)
  (ontable shot101)
  (dispenses dispenser368 ingredient153)
  (dispenses dispenser430 ingredient498)
  (dispenses dispenser389 ingredient3)
  (dispenses dispenser310 ingredient106)
  (clean shaker57)
  (clean shot10)
  (clean shot156)
  (clean shot101)
  (empty shaker57)
  (empty shot10)
  (empty shot156)
  (empty shot101)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker57 l0)
  (shaker-level shaker57 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient3)
  (cocktail-part2 cocktail1 ingredient498)
)
 (:goal
  (and
      (contains shot10 cocktail1)
      (contains shot156 cocktail1)
)))
