(define (problem prob)
 (:domain barman)
 (:objects 
      shaker403 - shaker
      left right - hand
      shot265 shot55 shot251 - shot
      ingredient233 ingredient48 ingredient104 - ingredient
      cocktail1 - cocktail
      dispenser495 dispenser293 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker403)
  (ontable shot265)
  (ontable shot55)
  (ontable shot251)
  (dispenses dispenser495 ingredient233)
  (dispenses dispenser293 ingredient48)
  (dispenses dispenser176 ingredient104)
  (clean shaker403)
  (clean shot265)
  (clean shot55)
  (clean shot251)
  (empty shaker403)
  (empty shot265)
  (empty shot55)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker403 l0)
  (shaker-level shaker403 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient233)
  (cocktail-part2 cocktail1 ingredient104)
)
 (:goal
  (and
      (contains shot265 cocktail1)
      (contains shot55 cocktail1)
)))
