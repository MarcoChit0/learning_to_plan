(define (problem prob)
 (:domain barman)
 (:objects 
      shaker443 - shaker
      left right - hand
      shot9 shot303 - shot
      ingredient462 ingredient448 ingredient483 - ingredient
      cocktail29 - cocktail
      dispenser203 dispenser340 dispenser220 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker443)
  (ontable shot9)
  (ontable shot303)
  (dispenses dispenser203 ingredient462)
  (dispenses dispenser340 ingredient448)
  (dispenses dispenser220 ingredient483)
  (clean shaker443)
  (clean shot9)
  (clean shot303)
  (empty shaker443)
  (empty shot9)
  (empty shot303)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker443 l0)
  (shaker-level shaker443 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail29 ingredient462)
  (cocktail-part2 cocktail29 ingredient448)
)
 (:goal
  (and
      (contains shot9 cocktail29)
)))
