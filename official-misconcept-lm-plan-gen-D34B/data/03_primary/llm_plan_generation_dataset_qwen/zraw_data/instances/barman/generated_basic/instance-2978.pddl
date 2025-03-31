(define (problem prob)
 (:domain barman)
 (:objects 
      shaker458 - shaker
      left right - hand
      shot195 - shot
      ingredient147 ingredient478 - ingredient
      cocktail4 - cocktail
      dispenser101 dispenser61 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker458)
  (ontable shot195)
  (dispenses dispenser101 ingredient147)
  (dispenses dispenser61 ingredient478)
  (clean shaker458)
  (clean shot195)
  (empty shaker458)
  (empty shot195)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker458 l0)
  (shaker-level shaker458 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail4 ingredient478)
  (cocktail-part2 cocktail4 ingredient147)
)
 (:goal
  (and
      (contains shot195 cocktail4)
)))
