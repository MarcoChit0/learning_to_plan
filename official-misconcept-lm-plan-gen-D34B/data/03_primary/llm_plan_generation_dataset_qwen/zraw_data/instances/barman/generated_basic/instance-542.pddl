(define (problem prob)
 (:domain barman)
 (:objects 
      shaker340 - shaker
      left right - hand
      shot251 - shot
      ingredient99 ingredient87 ingredient130 - ingredient
      cocktail1 - cocktail
      dispenser155 dispenser473 dispenser484 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker340)
  (ontable shot251)
  (dispenses dispenser155 ingredient99)
  (dispenses dispenser473 ingredient87)
  (dispenses dispenser484 ingredient130)
  (clean shaker340)
  (clean shot251)
  (empty shaker340)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker340 l0)
  (shaker-level shaker340 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient99)
  (cocktail-part2 cocktail1 ingredient130)
)
 (:goal
  (and
      (contains shot251 cocktail1)
)))
