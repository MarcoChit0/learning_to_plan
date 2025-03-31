(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot262 shot78 - shot
      ingredient159 ingredient68 ingredient385 ingredient365 - ingredient
      cocktail228 - cocktail
      dispenser386 dispenser439 dispenser224 dispenser145 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot262)
  (ontable shot78)
  (dispenses dispenser386 ingredient159)
  (dispenses dispenser439 ingredient68)
  (dispenses dispenser224 ingredient385)
  (dispenses dispenser145 ingredient365)
  (clean shaker126)
  (clean shot262)
  (clean shot78)
  (empty shaker126)
  (empty shot262)
  (empty shot78)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail228 ingredient68)
  (cocktail-part2 cocktail228 ingredient365)
)
 (:goal
  (and
      (contains shot262 cocktail228)
)))
