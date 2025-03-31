(define (problem prob)
 (:domain barman)
 (:objects 
      shaker43 - shaker
      left right - hand
      shot377 shot53 - shot
      ingredient350 ingredient182 - ingredient
      cocktail488 - cocktail
      dispenser289 dispenser313 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker43)
  (ontable shot377)
  (ontable shot53)
  (dispenses dispenser289 ingredient350)
  (dispenses dispenser313 ingredient182)
  (clean shaker43)
  (clean shot377)
  (clean shot53)
  (empty shaker43)
  (empty shot377)
  (empty shot53)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker43 l0)
  (shaker-level shaker43 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail488 ingredient350)
  (cocktail-part2 cocktail488 ingredient182)
)
 (:goal
  (and
      (contains shot377 cocktail488)
)))
