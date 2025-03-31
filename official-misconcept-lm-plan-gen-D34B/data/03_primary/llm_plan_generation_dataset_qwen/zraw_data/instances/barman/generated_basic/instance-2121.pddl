(define (problem prob)
 (:domain barman)
 (:objects 
      shaker56 - shaker
      left right - hand
      shot484 shot412 shot326 - shot
      ingredient384 ingredient90 - ingredient
      cocktail150 - cocktail
      dispenser477 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker56)
  (ontable shot484)
  (ontable shot412)
  (ontable shot326)
  (dispenses dispenser477 ingredient384)
  (dispenses dispenser43 ingredient90)
  (clean shaker56)
  (clean shot484)
  (clean shot412)
  (clean shot326)
  (empty shaker56)
  (empty shot484)
  (empty shot412)
  (empty shot326)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker56 l0)
  (shaker-level shaker56 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail150 ingredient384)
  (cocktail-part2 cocktail150 ingredient90)
)
 (:goal
  (and
      (contains shot484 cocktail150)
      (contains shot412 cocktail150)
)))
