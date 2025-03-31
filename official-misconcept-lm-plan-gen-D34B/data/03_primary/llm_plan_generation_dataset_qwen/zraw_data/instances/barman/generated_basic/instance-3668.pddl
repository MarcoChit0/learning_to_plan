(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot304 shot269 shot199 shot435 - shot
      ingredient488 ingredient154 ingredient209 - ingredient
      cocktail219 - cocktail
      dispenser411 dispenser309 dispenser265 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot304)
  (ontable shot269)
  (ontable shot199)
  (ontable shot435)
  (dispenses dispenser411 ingredient488)
  (dispenses dispenser309 ingredient154)
  (dispenses dispenser265 ingredient209)
  (clean shaker39)
  (clean shot304)
  (clean shot269)
  (clean shot199)
  (clean shot435)
  (empty shaker39)
  (empty shot304)
  (empty shot269)
  (empty shot199)
  (empty shot435)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail219 ingredient488)
  (cocktail-part2 cocktail219 ingredient154)
)
 (:goal
  (and
      (contains shot304 cocktail219)
      (contains shot269 ingredient209)
      (contains shot199 cocktail219)
)))
