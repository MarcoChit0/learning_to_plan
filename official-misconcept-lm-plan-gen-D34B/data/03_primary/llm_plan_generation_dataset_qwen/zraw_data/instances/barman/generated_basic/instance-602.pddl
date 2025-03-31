(define (problem prob)
 (:domain barman)
 (:objects 
      shaker154 - shaker
      left right - hand
      shot410 - shot
      ingredient227 ingredient280 ingredient491 ingredient234 - ingredient
      cocktail1 - cocktail
      dispenser15 dispenser496 dispenser257 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker154)
  (ontable shot410)
  (dispenses dispenser15 ingredient227)
  (dispenses dispenser496 ingredient280)
  (dispenses dispenser257 ingredient491)
  (dispenses dispenser166 ingredient234)
  (clean shaker154)
  (clean shot410)
  (empty shaker154)
  (empty shot410)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker154 l0)
  (shaker-level shaker154 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient234)
  (cocktail-part2 cocktail1 ingredient227)
)
 (:goal
  (and
      (contains shot410 cocktail1)
)))
