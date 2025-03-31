(define (problem prob)
 (:domain barman)
 (:objects 
      shaker212 - shaker
      left right - hand
      shot28 shot269 - shot
      ingredient229 ingredient312 ingredient44 ingredient319 - ingredient
      cocktail241 - cocktail
      dispenser96 dispenser387 dispenser191 dispenser229 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker212)
  (ontable shot28)
  (ontable shot269)
  (dispenses dispenser96 ingredient229)
  (dispenses dispenser387 ingredient312)
  (dispenses dispenser191 ingredient44)
  (dispenses dispenser229 ingredient319)
  (clean shaker212)
  (clean shot28)
  (clean shot269)
  (empty shaker212)
  (empty shot28)
  (empty shot269)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker212 l0)
  (shaker-level shaker212 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail241 ingredient312)
  (cocktail-part2 cocktail241 ingredient44)
)
 (:goal
  (and
      (contains shot28 cocktail241)
)))
