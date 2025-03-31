(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot397 shot102 shot48 - shot
      ingredient275 ingredient25 ingredient147 ingredient35 - ingredient
      cocktail53 - cocktail
      dispenser76 dispenser299 dispenser293 dispenser366 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot397)
  (ontable shot102)
  (ontable shot48)
  (dispenses dispenser76 ingredient275)
  (dispenses dispenser299 ingredient25)
  (dispenses dispenser293 ingredient147)
  (dispenses dispenser366 ingredient35)
  (clean shaker77)
  (clean shot397)
  (clean shot102)
  (clean shot48)
  (empty shaker77)
  (empty shot397)
  (empty shot102)
  (empty shot48)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail53 ingredient275)
  (cocktail-part2 cocktail53 ingredient147)
)
 (:goal
  (and
      (contains shot397 cocktail53)
      (contains shot102 cocktail53)
)))
