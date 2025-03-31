(define (problem prob)
 (:domain barman)
 (:objects 
      shaker204 - shaker
      left right - hand
      shot176 shot66 shot452 - shot
      ingredient366 ingredient274 ingredient87 - ingredient
      cocktail415 - cocktail
      dispenser21 dispenser343 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker204)
  (ontable shot176)
  (ontable shot66)
  (ontable shot452)
  (dispenses dispenser21 ingredient366)
  (dispenses dispenser343 ingredient274)
  (dispenses dispenser196 ingredient87)
  (clean shaker204)
  (clean shot176)
  (clean shot66)
  (clean shot452)
  (empty shaker204)
  (empty shot176)
  (empty shot66)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker204 l0)
  (shaker-level shaker204 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail415 ingredient87)
  (cocktail-part2 cocktail415 ingredient274)
)
 (:goal
  (and
      (contains shot176 cocktail415)
      (contains shot66 cocktail415)
)))
