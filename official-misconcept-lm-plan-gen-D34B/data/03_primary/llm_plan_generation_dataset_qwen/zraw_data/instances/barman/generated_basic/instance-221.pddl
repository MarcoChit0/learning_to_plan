(define (problem prob)
 (:domain barman)
 (:objects 
      shaker492 - shaker
      left right - hand
      shot74 - shot
      ingredient187 ingredient12 ingredient435 - ingredient
      cocktail1 - cocktail
      dispenser141 dispenser256 dispenser227 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker492)
  (ontable shot74)
  (dispenses dispenser141 ingredient187)
  (dispenses dispenser256 ingredient12)
  (dispenses dispenser227 ingredient435)
  (clean shaker492)
  (clean shot74)
  (empty shaker492)
  (empty shot74)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker492 l0)
  (shaker-level shaker492 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient12)
  (cocktail-part2 cocktail1 ingredient435)
)
 (:goal
  (and
      (contains shot74 cocktail1)
)))
