(define (problem prob)
 (:domain barman)
 (:objects 
      shaker412 - shaker
      left right - hand
      shot456 - shot
      ingredient305 ingredient184 ingredient497 - ingredient
      cocktail1 - cocktail
      dispenser372 dispenser30 dispenser105 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker412)
  (ontable shot456)
  (dispenses dispenser372 ingredient305)
  (dispenses dispenser30 ingredient184)
  (dispenses dispenser105 ingredient497)
  (clean shaker412)
  (clean shot456)
  (empty shaker412)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker412 l0)
  (shaker-level shaker412 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient305)
  (cocktail-part2 cocktail1 ingredient497)
)
 (:goal
  (and
      (contains shot456 cocktail1)
)))
