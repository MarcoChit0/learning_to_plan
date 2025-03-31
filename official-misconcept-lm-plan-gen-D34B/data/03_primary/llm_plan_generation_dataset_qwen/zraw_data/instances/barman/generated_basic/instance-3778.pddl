(define (problem prob)
 (:domain barman)
 (:objects 
      shaker68 - shaker
      left right - hand
      shot235 shot342 shot386 - shot
      ingredient192 ingredient41 - ingredient
      cocktail65 - cocktail
      dispenser290 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker68)
  (ontable shot235)
  (ontable shot342)
  (ontable shot386)
  (dispenses dispenser290 ingredient192)
  (dispenses dispenser57 ingredient41)
  (clean shaker68)
  (clean shot235)
  (clean shot342)
  (clean shot386)
  (empty shaker68)
  (empty shot235)
  (empty shot342)
  (empty shot386)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker68 l0)
  (shaker-level shaker68 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail65 ingredient192)
  (cocktail-part2 cocktail65 ingredient41)
)
 (:goal
  (and
      (contains shot235 cocktail65)
      (contains shot342 ingredient192)
)))
