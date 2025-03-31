(define (problem prob)
 (:domain barman)
 (:objects 
      shaker348 - shaker
      left right - hand
      shot123 - shot
      ingredient228 ingredient449 ingredient464 ingredient365 - ingredient
      cocktail435 - cocktail
      dispenser88 dispenser499 dispenser69 dispenser210 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker348)
  (ontable shot123)
  (dispenses dispenser88 ingredient228)
  (dispenses dispenser499 ingredient449)
  (dispenses dispenser69 ingredient464)
  (dispenses dispenser210 ingredient365)
  (clean shaker348)
  (clean shot123)
  (empty shaker348)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker348 l0)
  (shaker-level shaker348 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail435 ingredient464)
  (cocktail-part2 cocktail435 ingredient449)
)
 (:goal
  (and
      (contains shot123 cocktail435)
)))
