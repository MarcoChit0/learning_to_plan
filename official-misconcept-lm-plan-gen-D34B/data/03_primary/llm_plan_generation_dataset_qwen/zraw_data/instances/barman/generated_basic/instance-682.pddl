(define (problem prob)
 (:domain barman)
 (:objects 
      shaker377 - shaker
      left right - hand
      shot180 shot400 - shot
      ingredient111 ingredient18 - ingredient
      cocktail1 - cocktail
      dispenser58 dispenser302 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker377)
  (ontable shot180)
  (ontable shot400)
  (dispenses dispenser58 ingredient111)
  (dispenses dispenser302 ingredient18)
  (clean shaker377)
  (clean shot180)
  (clean shot400)
  (empty shaker377)
  (empty shot180)
  (empty shot400)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker377 l0)
  (shaker-level shaker377 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient111)
  (cocktail-part2 cocktail1 ingredient18)
)
 (:goal
  (and
      (contains shot180 cocktail1)
)))
