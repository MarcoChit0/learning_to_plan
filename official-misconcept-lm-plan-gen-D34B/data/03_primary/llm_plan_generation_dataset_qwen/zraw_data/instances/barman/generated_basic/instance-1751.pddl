(define (problem prob)
 (:domain barman)
 (:objects 
      shaker88 - shaker
      left right - hand
      shot376 shot377 - shot
      ingredient42 ingredient113 ingredient243 - ingredient
      cocktail14 - cocktail
      dispenser316 dispenser318 dispenser335 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker88)
  (ontable shot376)
  (ontable shot377)
  (dispenses dispenser316 ingredient42)
  (dispenses dispenser318 ingredient113)
  (dispenses dispenser335 ingredient243)
  (clean shaker88)
  (clean shot376)
  (clean shot377)
  (empty shaker88)
  (empty shot376)
  (empty shot377)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker88 l0)
  (shaker-level shaker88 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail14 ingredient42)
  (cocktail-part2 cocktail14 ingredient243)
)
 (:goal
  (and
      (contains shot376 cocktail14)
)))
