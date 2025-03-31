(define (problem prob)
 (:domain barman)
 (:objects 
      shaker329 - shaker
      left right - hand
      shot37 shot375 - shot
      ingredient68 ingredient354 - ingredient
      cocktail1 - cocktail
      dispenser191 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker329)
  (ontable shot37)
  (ontable shot375)
  (dispenses dispenser191 ingredient68)
  (dispenses dispenser342 ingredient354)
  (clean shaker329)
  (clean shot37)
  (clean shot375)
  (empty shaker329)
  (empty shot37)
  (empty shot375)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker329 l0)
  (shaker-level shaker329 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient68)
  (cocktail-part2 cocktail1 ingredient354)
)
 (:goal
  (and
      (contains shot37 cocktail1)
)))
