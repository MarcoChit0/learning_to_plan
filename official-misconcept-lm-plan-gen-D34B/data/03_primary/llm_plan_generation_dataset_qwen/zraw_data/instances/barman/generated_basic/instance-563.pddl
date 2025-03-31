(define (problem prob)
 (:domain barman)
 (:objects 
      shaker298 - shaker
      left right - hand
      shot282 shot359 - shot
      ingredient125 ingredient187 ingredient300 - ingredient
      cocktail1 - cocktail
      dispenser330 dispenser149 dispenser28 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker298)
  (ontable shot282)
  (ontable shot359)
  (dispenses dispenser330 ingredient125)
  (dispenses dispenser149 ingredient187)
  (dispenses dispenser28 ingredient300)
  (clean shaker298)
  (clean shot282)
  (clean shot359)
  (empty shaker298)
  (empty shot282)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker298 l0)
  (shaker-level shaker298 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient125)
  (cocktail-part2 cocktail1 ingredient187)
)
 (:goal
  (and
      (contains shot282 cocktail1)
)))
