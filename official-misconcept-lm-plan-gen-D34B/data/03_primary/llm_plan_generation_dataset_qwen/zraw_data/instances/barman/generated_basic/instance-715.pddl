(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot332 - shot
      ingredient187 ingredient94 ingredient302 - ingredient
      cocktail1 - cocktail
      dispenser18 dispenser458 dispenser89 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot332)
  (dispenses dispenser18 ingredient187)
  (dispenses dispenser458 ingredient94)
  (dispenses dispenser89 ingredient302)
  (clean shaker315)
  (clean shot332)
  (empty shaker315)
  (empty shot332)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient94)
  (cocktail-part2 cocktail1 ingredient187)
)
 (:goal
  (and
      (contains shot332 cocktail1)
)))
