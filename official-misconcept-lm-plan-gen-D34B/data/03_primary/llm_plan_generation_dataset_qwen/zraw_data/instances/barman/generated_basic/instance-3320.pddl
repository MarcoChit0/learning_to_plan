(define (problem prob)
 (:domain barman)
 (:objects 
      shaker266 - shaker
      left right - hand
      shot471 shot286 shot5 - shot
      ingredient160 ingredient446 ingredient251 - ingredient
      cocktail206 - cocktail
      dispenser233 dispenser281 dispenser8 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker266)
  (ontable shot471)
  (ontable shot286)
  (ontable shot5)
  (dispenses dispenser233 ingredient160)
  (dispenses dispenser281 ingredient446)
  (dispenses dispenser8 ingredient251)
  (clean shaker266)
  (clean shot471)
  (clean shot286)
  (clean shot5)
  (empty shaker266)
  (empty shot471)
  (empty shot286)
  (empty shot5)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker266 l0)
  (shaker-level shaker266 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail206 ingredient160)
  (cocktail-part2 cocktail206 ingredient446)
)
 (:goal
  (and
      (contains shot471 cocktail206)
      (contains shot286 cocktail206)
)))
