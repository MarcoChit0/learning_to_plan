(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot397 shot169 - shot
      ingredient269 ingredient287 ingredient444 ingredient157 - ingredient
      cocktail306 - cocktail
      dispenser412 dispenser303 dispenser132 dispenser64 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot397)
  (ontable shot169)
  (dispenses dispenser412 ingredient269)
  (dispenses dispenser303 ingredient287)
  (dispenses dispenser132 ingredient444)
  (dispenses dispenser64 ingredient157)
  (clean shaker294)
  (clean shot397)
  (clean shot169)
  (empty shaker294)
  (empty shot397)
  (empty shot169)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail306 ingredient444)
  (cocktail-part2 cocktail306 ingredient269)
)
 (:goal
  (and
      (contains shot397 cocktail306)
)))
