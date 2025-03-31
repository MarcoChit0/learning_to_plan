(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot412 shot478 - shot
      ingredient424 ingredient208 ingredient497 ingredient51 - ingredient
      cocktail1 - cocktail
      dispenser266 dispenser471 dispenser96 dispenser125 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot412)
  (ontable shot478)
  (dispenses dispenser266 ingredient424)
  (dispenses dispenser471 ingredient208)
  (dispenses dispenser96 ingredient497)
  (dispenses dispenser125 ingredient51)
  (clean shaker151)
  (clean shot412)
  (clean shot478)
  (empty shaker151)
  (empty shot412)
  (empty shot478)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient424)
  (cocktail-part2 cocktail1 ingredient51)
)
 (:goal
  (and
      (contains shot412 cocktail1)
)))
