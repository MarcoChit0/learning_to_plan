(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot74 shot208 shot260 - shot
      ingredient425 ingredient92 - ingredient
      cocktail71 - cocktail
      dispenser303 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot74)
  (ontable shot208)
  (ontable shot260)
  (dispenses dispenser303 ingredient425)
  (dispenses dispenser293 ingredient92)
  (clean shaker39)
  (clean shot74)
  (clean shot208)
  (clean shot260)
  (empty shaker39)
  (empty shot74)
  (empty shot208)
  (empty shot260)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail71 ingredient92)
  (cocktail-part2 cocktail71 ingredient425)
)
 (:goal
  (and
      (contains shot74 cocktail71)
      (contains shot208 ingredient92)
)))
