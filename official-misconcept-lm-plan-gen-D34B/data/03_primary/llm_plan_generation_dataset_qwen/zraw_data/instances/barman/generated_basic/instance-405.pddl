(define (problem prob)
 (:domain barman)
 (:objects 
      shaker4 - shaker
      left right - hand
      shot122 shot263 - shot
      ingredient54 ingredient254 - ingredient
      cocktail1 - cocktail
      dispenser289 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker4)
  (ontable shot122)
  (ontable shot263)
  (dispenses dispenser289 ingredient54)
  (dispenses dispenser262 ingredient254)
  (clean shaker4)
  (clean shot122)
  (clean shot263)
  (empty shaker4)
  (empty shot122)
  (empty shot263)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker4 l0)
  (shaker-level shaker4 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient54)
  (cocktail-part2 cocktail1 ingredient254)
)
 (:goal
  (and
      (contains shot122 cocktail1)
)))
