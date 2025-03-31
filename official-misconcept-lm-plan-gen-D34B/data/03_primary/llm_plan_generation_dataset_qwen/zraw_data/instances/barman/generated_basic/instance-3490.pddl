(define (problem prob)
 (:domain barman)
 (:objects 
      shaker383 - shaker
      left right - hand
      shot176 shot18 shot30 - shot
      ingredient232 ingredient219 ingredient79 - ingredient
      cocktail242 - cocktail
      dispenser136 dispenser353 dispenser106 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker383)
  (ontable shot176)
  (ontable shot18)
  (ontable shot30)
  (dispenses dispenser136 ingredient232)
  (dispenses dispenser353 ingredient219)
  (dispenses dispenser106 ingredient79)
  (clean shaker383)
  (clean shot176)
  (clean shot18)
  (clean shot30)
  (empty shaker383)
  (empty shot176)
  (empty shot18)
  (empty shot30)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker383 l0)
  (shaker-level shaker383 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail242 ingredient79)
  (cocktail-part2 cocktail242 ingredient232)
)
 (:goal
  (and
      (contains shot176 cocktail242)
      (contains shot18 ingredient79)
)))
