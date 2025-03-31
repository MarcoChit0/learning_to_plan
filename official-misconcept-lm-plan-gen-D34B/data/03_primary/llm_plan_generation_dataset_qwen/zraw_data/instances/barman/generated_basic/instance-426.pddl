(define (problem prob)
 (:domain barman)
 (:objects 
      shaker269 - shaker
      left right - hand
      shot91 - shot
      ingredient176 ingredient48 - ingredient
      cocktail1 - cocktail
      dispenser304 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker269)
  (ontable shot91)
  (dispenses dispenser304 ingredient176)
  (dispenses dispenser429 ingredient48)
  (clean shaker269)
  (clean shot91)
  (empty shaker269)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker269 l0)
  (shaker-level shaker269 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient48)
  (cocktail-part2 cocktail1 ingredient176)
)
 (:goal
  (and
      (contains shot91 cocktail1)
)))
