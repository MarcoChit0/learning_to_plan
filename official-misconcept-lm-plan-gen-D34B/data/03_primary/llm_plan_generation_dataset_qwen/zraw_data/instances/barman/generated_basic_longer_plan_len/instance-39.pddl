(define (problem prob)
 (:domain barman)
 (:objects 
      shaker324 - shaker
      left right - hand
      shot14 shot480 shot415 - shot
      ingredient401 ingredient4 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser22 dispenser155 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker324)
  (ontable shot14)
  (ontable shot480)
  (ontable shot415)
  (dispenses dispenser22 ingredient401)
  (dispenses dispenser155 ingredient4)
  (clean shaker324)
  (clean shot14)
  (clean shot480)
  (clean shot415)
  (empty shaker324)
  (empty shot14)
  (empty shot480)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker324 l0)
  (shaker-level shaker324 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient401)
  (cocktail-part2 cocktail1 ingredient4)
  (cocktail-part1 cocktail2 ingredient4)
  (cocktail-part2 cocktail2 ingredient401)
)
 (:goal
  (and
      (contains shot14 cocktail1)
      (contains shot480 cocktail2)
)))
