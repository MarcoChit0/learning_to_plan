(define (problem prob)
 (:domain barman)
 (:objects 
      shaker133 - shaker
      left right - hand
      shot111 shot160 - shot
      ingredient271 ingredient41 ingredient390 - ingredient
      cocktail1 - cocktail
      dispenser254 dispenser265 dispenser22 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker133)
  (ontable shot111)
  (ontable shot160)
  (dispenses dispenser254 ingredient271)
  (dispenses dispenser265 ingredient41)
  (dispenses dispenser22 ingredient390)
  (clean shaker133)
  (clean shot111)
  (clean shot160)
  (empty shaker133)
  (empty shot111)
  (empty shot160)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker133 l0)
  (shaker-level shaker133 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient271)
  (cocktail-part2 cocktail1 ingredient41)
)
 (:goal
  (and
      (contains shot111 cocktail1)
)))
