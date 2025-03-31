(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot375 shot396 shot191 - shot
      ingredient192 ingredient109 - ingredient
      cocktail386 - cocktail
      dispenser446 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot375)
  (ontable shot396)
  (ontable shot191)
  (dispenses dispenser446 ingredient192)
  (dispenses dispenser292 ingredient109)
  (clean shaker241)
  (clean shot375)
  (clean shot396)
  (clean shot191)
  (empty shaker241)
  (empty shot375)
  (empty shot396)
  (empty shot191)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail386 ingredient109)
  (cocktail-part2 cocktail386 ingredient192)
)
 (:goal
  (and
      (contains shot375 cocktail386)
      (contains shot396 cocktail386)
)))
