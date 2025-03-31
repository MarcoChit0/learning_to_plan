(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot254 shot101 shot107 - shot
      ingredient16 ingredient100 ingredient489 - ingredient
      cocktail1 - cocktail
      dispenser73 dispenser437 dispenser424 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot254)
  (ontable shot101)
  (ontable shot107)
  (dispenses dispenser73 ingredient16)
  (dispenses dispenser437 ingredient100)
  (dispenses dispenser424 ingredient489)
  (clean shaker22)
  (clean shot254)
  (clean shot101)
  (clean shot107)
  (empty shaker22)
  (empty shot254)
  (empty shot101)
  (empty shot107)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient489)
  (cocktail-part2 cocktail1 ingredient16)
)
 (:goal
  (and
      (contains shot254 cocktail1)
      (contains shot101 cocktail1)
)))
