(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot92 - shot
      ingredient20 ingredient250 ingredient435 - ingredient
      cocktail1 - cocktail
      dispenser478 dispenser291 dispenser270 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot92)
  (dispenses dispenser478 ingredient20)
  (dispenses dispenser291 ingredient250)
  (dispenses dispenser270 ingredient435)
  (clean shaker380)
  (clean shot92)
  (empty shaker380)
  (empty shot92)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient20)
  (cocktail-part2 cocktail1 ingredient250)
)
 (:goal
  (and
      (contains shot92 cocktail1)
)))
