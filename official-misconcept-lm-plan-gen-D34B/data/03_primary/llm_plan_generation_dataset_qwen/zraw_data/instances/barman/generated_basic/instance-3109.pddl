(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot8 shot480 - shot
      ingredient254 ingredient267 - ingredient
      cocktail110 - cocktail
      dispenser72 dispenser366 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot8)
  (ontable shot480)
  (dispenses dispenser72 ingredient254)
  (dispenses dispenser366 ingredient267)
  (clean shaker303)
  (clean shot8)
  (clean shot480)
  (empty shaker303)
  (empty shot8)
  (empty shot480)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail110 ingredient254)
  (cocktail-part2 cocktail110 ingredient267)
)
 (:goal
  (and
      (contains shot8 cocktail110)
)))
