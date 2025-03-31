(define (problem prob)
 (:domain barman)
 (:objects 
      shaker423 - shaker
      left right - hand
      shot251 shot171 - shot
      ingredient369 ingredient238 ingredient443 ingredient335 - ingredient
      cocktail49 - cocktail
      dispenser104 dispenser134 dispenser486 dispenser437 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker423)
  (ontable shot251)
  (ontable shot171)
  (dispenses dispenser104 ingredient369)
  (dispenses dispenser134 ingredient238)
  (dispenses dispenser486 ingredient443)
  (dispenses dispenser437 ingredient335)
  (clean shaker423)
  (clean shot251)
  (clean shot171)
  (empty shaker423)
  (empty shot251)
  (empty shot171)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker423 l0)
  (shaker-level shaker423 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail49 ingredient335)
  (cocktail-part2 cocktail49 ingredient369)
)
 (:goal
  (and
      (contains shot251 cocktail49)
)))
