(define (problem prob)
 (:domain barman)
 (:objects 
      shaker204 - shaker
      left right - hand
      shot142 - shot
      ingredient227 ingredient308 ingredient289 - ingredient
      cocktail355 - cocktail
      dispenser74 dispenser124 dispenser135 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker204)
  (ontable shot142)
  (dispenses dispenser74 ingredient227)
  (dispenses dispenser124 ingredient308)
  (dispenses dispenser135 ingredient289)
  (clean shaker204)
  (clean shot142)
  (empty shaker204)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker204 l0)
  (shaker-level shaker204 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail355 ingredient227)
  (cocktail-part2 cocktail355 ingredient289)
)
 (:goal
  (and
      (contains shot142 cocktail355)
)))
