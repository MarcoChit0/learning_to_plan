(define (problem prob)
 (:domain barman)
 (:objects 
      shaker296 - shaker
      left right - hand
      shot254 shot401 shot180 - shot
      ingredient27 ingredient82 ingredient106 ingredient133 - ingredient
      cocktail1 - cocktail
      dispenser125 dispenser104 dispenser462 dispenser290 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker296)
  (ontable shot254)
  (ontable shot401)
  (ontable shot180)
  (dispenses dispenser125 ingredient27)
  (dispenses dispenser104 ingredient82)
  (dispenses dispenser462 ingredient106)
  (dispenses dispenser290 ingredient133)
  (clean shaker296)
  (clean shot254)
  (clean shot401)
  (clean shot180)
  (empty shaker296)
  (empty shot254)
  (empty shot401)
  (empty shot180)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker296 l0)
  (shaker-level shaker296 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient106)
  (cocktail-part2 cocktail1 ingredient82)
)
 (:goal
  (and
      (contains shot254 cocktail1)
      (contains shot401 ingredient106)
)))
