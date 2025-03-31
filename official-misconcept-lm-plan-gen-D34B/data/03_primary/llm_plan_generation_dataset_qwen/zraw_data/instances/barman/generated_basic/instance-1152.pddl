(define (problem prob)
 (:domain barman)
 (:objects 
      shaker68 - shaker
      left right - hand
      shot270 shot126 - shot
      ingredient246 ingredient401 ingredient418 ingredient2 - ingredient
      cocktail1 - cocktail
      dispenser349 dispenser186 dispenser464 dispenser365 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker68)
  (ontable shot270)
  (ontable shot126)
  (dispenses dispenser349 ingredient246)
  (dispenses dispenser186 ingredient401)
  (dispenses dispenser464 ingredient418)
  (dispenses dispenser365 ingredient2)
  (clean shaker68)
  (clean shot270)
  (clean shot126)
  (empty shaker68)
  (empty shot270)
  (empty shot126)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker68 l0)
  (shaker-level shaker68 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient401)
  (cocktail-part2 cocktail1 ingredient246)
)
 (:goal
  (and
      (contains shot270 cocktail1)
)))
