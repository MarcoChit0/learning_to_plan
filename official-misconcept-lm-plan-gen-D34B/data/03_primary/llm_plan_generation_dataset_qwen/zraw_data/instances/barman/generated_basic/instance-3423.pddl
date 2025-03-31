(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot256 shot144 - shot
      ingredient431 ingredient131 ingredient241 - ingredient
      cocktail366 - cocktail
      dispenser375 dispenser248 dispenser54 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot256)
  (ontable shot144)
  (dispenses dispenser375 ingredient431)
  (dispenses dispenser248 ingredient131)
  (dispenses dispenser54 ingredient241)
  (clean shaker360)
  (clean shot256)
  (clean shot144)
  (empty shaker360)
  (empty shot256)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail366 ingredient131)
  (cocktail-part2 cocktail366 ingredient431)
)
 (:goal
  (and
      (contains shot256 cocktail366)
)))
