(define (problem prob)
 (:domain barman)
 (:objects 
      shaker272 - shaker
      left right - hand
      shot219 - shot
      ingredient223 ingredient447 ingredient435 ingredient227 - ingredient
      cocktail5 - cocktail
      dispenser399 dispenser89 dispenser355 dispenser120 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker272)
  (ontable shot219)
  (dispenses dispenser399 ingredient223)
  (dispenses dispenser89 ingredient447)
  (dispenses dispenser355 ingredient435)
  (dispenses dispenser120 ingredient227)
  (clean shaker272)
  (clean shot219)
  (empty shaker272)
  (empty shot219)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker272 l0)
  (shaker-level shaker272 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail5 ingredient435)
  (cocktail-part2 cocktail5 ingredient227)
)
 (:goal
  (and
      (contains shot219 cocktail5)
)))
