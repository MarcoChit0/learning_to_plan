(define (problem prob)
 (:domain barman)
 (:objects 
      shaker123 - shaker
      left right - hand
      shot105 - shot
      ingredient360 ingredient239 ingredient157 ingredient14 - ingredient
      cocktail1 - cocktail
      dispenser362 dispenser307 dispenser280 dispenser8 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker123)
  (ontable shot105)
  (dispenses dispenser362 ingredient360)
  (dispenses dispenser307 ingredient239)
  (dispenses dispenser280 ingredient157)
  (dispenses dispenser8 ingredient14)
  (clean shaker123)
  (clean shot105)
  (empty shaker123)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker123 l0)
  (shaker-level shaker123 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient14)
  (cocktail-part2 cocktail1 ingredient157)
)
 (:goal
  (and
      (contains shot105 cocktail1)
)))
