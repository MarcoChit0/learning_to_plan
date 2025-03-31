(define (problem prob)
 (:domain barman)
 (:objects 
      shaker41 - shaker
      left right - hand
      shot430 - shot
      ingredient251 ingredient36 ingredient498 - ingredient
      cocktail1 - cocktail
      dispenser138 dispenser454 dispenser68 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker41)
  (ontable shot430)
  (dispenses dispenser138 ingredient251)
  (dispenses dispenser454 ingredient36)
  (dispenses dispenser68 ingredient498)
  (clean shaker41)
  (clean shot430)
  (empty shaker41)
  (empty shot430)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker41 l0)
  (shaker-level shaker41 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient498)
  (cocktail-part2 cocktail1 ingredient251)
)
 (:goal
  (and
      (contains shot430 cocktail1)
)))
