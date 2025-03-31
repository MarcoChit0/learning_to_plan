(define (problem prob)
 (:domain barman)
 (:objects 
      shaker377 - shaker
      left right - hand
      shot439 shot422 - shot
      ingredient267 ingredient135 ingredient227 - ingredient
      cocktail1 - cocktail
      dispenser167 dispenser149 dispenser164 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker377)
  (ontable shot439)
  (ontable shot422)
  (dispenses dispenser167 ingredient267)
  (dispenses dispenser149 ingredient135)
  (dispenses dispenser164 ingredient227)
  (clean shaker377)
  (clean shot439)
  (clean shot422)
  (empty shaker377)
  (empty shot439)
  (empty shot422)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker377 l0)
  (shaker-level shaker377 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient267)
  (cocktail-part2 cocktail1 ingredient135)
)
 (:goal
  (and
      (contains shot439 cocktail1)
)))
