(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot437 shot201 - shot
      ingredient357 ingredient424 ingredient280 - ingredient
      cocktail191 - cocktail
      dispenser134 dispenser80 dispenser210 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot437)
  (ontable shot201)
  (dispenses dispenser134 ingredient357)
  (dispenses dispenser80 ingredient424)
  (dispenses dispenser210 ingredient280)
  (clean shaker350)
  (clean shot437)
  (clean shot201)
  (empty shaker350)
  (empty shot437)
  (empty shot201)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail191 ingredient424)
  (cocktail-part2 cocktail191 ingredient357)
)
 (:goal
  (and
      (contains shot437 cocktail191)
)))
