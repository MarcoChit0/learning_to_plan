(define (problem prob)
 (:domain barman)
 (:objects 
      shaker355 - shaker
      left right - hand
      shot175 - shot
      ingredient51 ingredient68 - ingredient
      cocktail51 - cocktail
      dispenser268 dispenser90 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker355)
  (ontable shot175)
  (dispenses dispenser268 ingredient51)
  (dispenses dispenser90 ingredient68)
  (clean shaker355)
  (clean shot175)
  (empty shaker355)
  (empty shot175)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker355 l0)
  (shaker-level shaker355 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail51 ingredient68)
  (cocktail-part2 cocktail51 ingredient51)
)
 (:goal
  (and
      (contains shot175 cocktail51)
)))
