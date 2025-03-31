(define (problem prob)
 (:domain barman)
 (:objects 
      shaker496 - shaker
      left right - hand
      shot40 shot440 shot94 - shot
      ingredient397 ingredient92 ingredient267 ingredient87 - ingredient
      cocktail185 - cocktail
      dispenser475 dispenser397 dispenser347 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker496)
  (ontable shot40)
  (ontable shot440)
  (ontable shot94)
  (dispenses dispenser475 ingredient397)
  (dispenses dispenser397 ingredient92)
  (dispenses dispenser347 ingredient267)
  (dispenses dispenser152 ingredient87)
  (clean shaker496)
  (clean shot40)
  (clean shot440)
  (clean shot94)
  (empty shaker496)
  (empty shot40)
  (empty shot440)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker496 l0)
  (shaker-level shaker496 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail185 ingredient267)
  (cocktail-part2 cocktail185 ingredient87)
)
 (:goal
  (and
      (contains shot40 cocktail185)
      (contains shot440 ingredient87)
)))
