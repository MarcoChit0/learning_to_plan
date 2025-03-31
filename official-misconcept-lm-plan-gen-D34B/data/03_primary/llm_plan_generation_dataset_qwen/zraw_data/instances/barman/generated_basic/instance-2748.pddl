(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot370 - shot
      ingredient324 ingredient365 - ingredient
      cocktail28 - cocktail
      dispenser358 dispenser153 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot370)
  (dispenses dispenser358 ingredient324)
  (dispenses dispenser153 ingredient365)
  (clean shaker170)
  (clean shot370)
  (empty shaker170)
  (empty shot370)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail28 ingredient365)
  (cocktail-part2 cocktail28 ingredient324)
)
 (:goal
  (and
      (contains shot370 cocktail28)
)))
