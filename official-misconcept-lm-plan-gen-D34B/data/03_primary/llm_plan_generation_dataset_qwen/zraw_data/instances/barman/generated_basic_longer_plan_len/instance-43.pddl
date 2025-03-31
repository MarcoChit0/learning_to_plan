(define (problem prob)
 (:domain barman)
 (:objects 
      shaker285 - shaker
      left right - hand
      shot176 shot12 shot70 - shot
      ingredient402 ingredient133 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser41 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker285)
  (ontable shot176)
  (ontable shot12)
  (ontable shot70)
  (dispenses dispenser41 ingredient402)
  (dispenses dispenser296 ingredient133)
  (clean shaker285)
  (clean shot176)
  (clean shot12)
  (clean shot70)
  (empty shaker285)
  (empty shot176)
  (empty shot12)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker285 l0)
  (shaker-level shaker285 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient133)
  (cocktail-part2 cocktail1 ingredient402)
  (cocktail-part1 cocktail2 ingredient402)
  (cocktail-part2 cocktail2 ingredient133)
)
 (:goal
  (and
      (contains shot176 cocktail2)
      (contains shot12 cocktail1)
)))
