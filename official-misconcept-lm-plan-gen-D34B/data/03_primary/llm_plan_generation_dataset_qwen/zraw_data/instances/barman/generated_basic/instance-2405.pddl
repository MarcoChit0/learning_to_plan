(define (problem prob)
 (:domain barman)
 (:objects 
      shaker193 - shaker
      left right - hand
      shot77 - shot
      ingredient178 ingredient242 - ingredient
      cocktail57 - cocktail
      dispenser410 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker193)
  (ontable shot77)
  (dispenses dispenser410 ingredient178)
  (dispenses dispenser474 ingredient242)
  (clean shaker193)
  (clean shot77)
  (empty shaker193)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker193 l0)
  (shaker-level shaker193 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail57 ingredient242)
  (cocktail-part2 cocktail57 ingredient178)
)
 (:goal
  (and
      (contains shot77 cocktail57)
)))
