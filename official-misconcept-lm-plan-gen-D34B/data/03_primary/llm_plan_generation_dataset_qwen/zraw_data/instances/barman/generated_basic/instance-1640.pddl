(define (problem prob)
 (:domain barman)
 (:objects 
      shaker397 - shaker
      left right - hand
      shot142 shot412 shot198 - shot
      ingredient23 ingredient56 ingredient266 ingredient172 - ingredient
      cocktail178 - cocktail
      dispenser149 dispenser304 dispenser66 dispenser236 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker397)
  (ontable shot142)
  (ontable shot412)
  (ontable shot198)
  (dispenses dispenser149 ingredient23)
  (dispenses dispenser304 ingredient56)
  (dispenses dispenser66 ingredient266)
  (dispenses dispenser236 ingredient172)
  (clean shaker397)
  (clean shot142)
  (clean shot412)
  (clean shot198)
  (empty shaker397)
  (empty shot142)
  (empty shot412)
  (empty shot198)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker397 l0)
  (shaker-level shaker397 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail178 ingredient23)
  (cocktail-part2 cocktail178 ingredient56)
)
 (:goal
  (and
      (contains shot142 cocktail178)
      (contains shot412 cocktail178)
)))
