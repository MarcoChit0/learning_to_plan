(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot281 shot292 - shot
      ingredient234 ingredient451 - ingredient
      cocktail1 - cocktail
      dispenser494 dispenser53 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot281)
  (ontable shot292)
  (dispenses dispenser494 ingredient234)
  (dispenses dispenser53 ingredient451)
  (clean shaker221)
  (clean shot281)
  (clean shot292)
  (empty shaker221)
  (empty shot281)
  (empty shot292)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient234)
  (cocktail-part2 cocktail1 ingredient451)
)
 (:goal
  (and
      (contains shot281 cocktail1)
)))
