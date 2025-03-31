(define (problem prob)
 (:domain barman)
 (:objects 
      shaker487 - shaker
      left right - hand
      shot40 shot154 - shot
      ingredient417 ingredient294 ingredient244 - ingredient
      cocktail1 - cocktail
      dispenser284 dispenser428 dispenser419 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker487)
  (ontable shot40)
  (ontable shot154)
  (dispenses dispenser284 ingredient417)
  (dispenses dispenser428 ingredient294)
  (dispenses dispenser419 ingredient244)
  (clean shaker487)
  (clean shot40)
  (clean shot154)
  (empty shaker487)
  (empty shot40)
  (empty shot154)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker487 l0)
  (shaker-level shaker487 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient417)
  (cocktail-part2 cocktail1 ingredient244)
)
 (:goal
  (and
      (contains shot40 cocktail1)
)))
