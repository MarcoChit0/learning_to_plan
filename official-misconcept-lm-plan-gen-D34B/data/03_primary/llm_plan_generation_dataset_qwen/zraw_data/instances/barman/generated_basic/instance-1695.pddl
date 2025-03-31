(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot348 - shot
      ingredient35 ingredient153 ingredient106 ingredient302 - ingredient
      cocktail449 - cocktail
      dispenser60 dispenser280 dispenser479 dispenser61 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot348)
  (dispenses dispenser60 ingredient35)
  (dispenses dispenser280 ingredient153)
  (dispenses dispenser479 ingredient106)
  (dispenses dispenser61 ingredient302)
  (clean shaker498)
  (clean shot348)
  (empty shaker498)
  (empty shot348)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail449 ingredient153)
  (cocktail-part2 cocktail449 ingredient35)
)
 (:goal
  (and
      (contains shot348 cocktail449)
)))
