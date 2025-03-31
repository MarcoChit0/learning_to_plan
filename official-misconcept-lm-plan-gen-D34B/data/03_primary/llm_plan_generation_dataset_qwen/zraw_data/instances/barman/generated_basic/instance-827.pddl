(define (problem prob)
 (:domain barman)
 (:objects 
      shaker102 - shaker
      left right - hand
      shot186 shot284 shot170 - shot
      ingredient305 ingredient222 ingredient81 ingredient269 - ingredient
      cocktail1 - cocktail
      dispenser353 dispenser173 dispenser271 dispenser0 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker102)
  (ontable shot186)
  (ontable shot284)
  (ontable shot170)
  (dispenses dispenser353 ingredient305)
  (dispenses dispenser173 ingredient222)
  (dispenses dispenser271 ingredient81)
  (dispenses dispenser0 ingredient269)
  (clean shaker102)
  (clean shot186)
  (clean shot284)
  (clean shot170)
  (empty shaker102)
  (empty shot186)
  (empty shot284)
  (empty shot170)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker102 l0)
  (shaker-level shaker102 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient269)
  (cocktail-part2 cocktail1 ingredient305)
)
 (:goal
  (and
      (contains shot186 cocktail1)
      (contains shot284 cocktail1)
)))
