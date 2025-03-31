(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot487 shot121 shot302 - shot
      ingredient334 ingredient375 ingredient46 ingredient170 - ingredient
      cocktail68 - cocktail
      dispenser113 dispenser265 dispenser284 dispenser96 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot487)
  (ontable shot121)
  (ontable shot302)
  (dispenses dispenser113 ingredient334)
  (dispenses dispenser265 ingredient375)
  (dispenses dispenser284 ingredient46)
  (dispenses dispenser96 ingredient170)
  (clean shaker498)
  (clean shot487)
  (clean shot121)
  (clean shot302)
  (empty shaker498)
  (empty shot487)
  (empty shot121)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail68 ingredient334)
  (cocktail-part2 cocktail68 ingredient170)
)
 (:goal
  (and
      (contains shot487 cocktail68)
      (contains shot121 ingredient375)
)))
