(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot255 shot126 - shot
      ingredient454 ingredient49 ingredient149 ingredient98 - ingredient
      cocktail122 - cocktail
      dispenser45 dispenser389 dispenser41 dispenser404 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot255)
  (ontable shot126)
  (dispenses dispenser45 ingredient454)
  (dispenses dispenser389 ingredient49)
  (dispenses dispenser41 ingredient149)
  (dispenses dispenser404 ingredient98)
  (clean shaker1)
  (clean shot255)
  (clean shot126)
  (empty shaker1)
  (empty shot255)
  (empty shot126)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker1 l0)
  (shaker-level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail122 ingredient149)
  (cocktail-part2 cocktail122 ingredient98)
)
 (:goal
  (and
      (contains shot255 cocktail122)
)))
