(define (problem prob)
 (:domain barman)
 (:objects 
      shaker209 - shaker
      left right - hand
      shot49 shot52 shot85 - shot
      ingredient479 ingredient160 - ingredient
      cocktail100 - cocktail
      dispenser479 dispenser278 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker209)
  (ontable shot49)
  (ontable shot52)
  (ontable shot85)
  (dispenses dispenser479 ingredient479)
  (dispenses dispenser278 ingredient160)
  (clean shaker209)
  (clean shot49)
  (clean shot52)
  (clean shot85)
  (empty shaker209)
  (empty shot49)
  (empty shot52)
  (empty shot85)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker209 l0)
  (shaker-level shaker209 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail100 ingredient479)
  (cocktail-part2 cocktail100 ingredient160)
)
 (:goal
  (and
      (contains shot49 cocktail100)
      (contains shot52 ingredient479)
)))
