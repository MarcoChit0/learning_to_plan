(define (problem prob)
 (:domain barman)
 (:objects 
      shaker254 - shaker
      left right - hand
      shot270 shot221 shot208 - shot
      ingredient65 ingredient287 - ingredient
      cocktail126 - cocktail
      dispenser316 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker254)
  (ontable shot270)
  (ontable shot221)
  (ontable shot208)
  (dispenses dispenser316 ingredient65)
  (dispenses dispenser130 ingredient287)
  (clean shaker254)
  (clean shot270)
  (clean shot221)
  (clean shot208)
  (empty shaker254)
  (empty shot270)
  (empty shot221)
  (empty shot208)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker254 l0)
  (shaker-level shaker254 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail126 ingredient287)
  (cocktail-part2 cocktail126 ingredient65)
)
 (:goal
  (and
      (contains shot270 cocktail126)
      (contains shot221 ingredient287)
)))
