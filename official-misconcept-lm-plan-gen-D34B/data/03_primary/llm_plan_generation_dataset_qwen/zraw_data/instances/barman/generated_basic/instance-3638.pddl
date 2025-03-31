(define (problem prob)
 (:domain barman)
 (:objects 
      shaker370 - shaker
      left right - hand
      shot454 - shot
      ingredient43 ingredient270 ingredient237 ingredient339 - ingredient
      cocktail473 - cocktail
      dispenser8 dispenser15 dispenser342 dispenser258 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker370)
  (ontable shot454)
  (dispenses dispenser8 ingredient43)
  (dispenses dispenser15 ingredient270)
  (dispenses dispenser342 ingredient237)
  (dispenses dispenser258 ingredient339)
  (clean shaker370)
  (clean shot454)
  (empty shaker370)
  (empty shot454)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker370 l0)
  (shaker-level shaker370 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail473 ingredient270)
  (cocktail-part2 cocktail473 ingredient339)
)
 (:goal
  (and
      (contains shot454 cocktail473)
)))
