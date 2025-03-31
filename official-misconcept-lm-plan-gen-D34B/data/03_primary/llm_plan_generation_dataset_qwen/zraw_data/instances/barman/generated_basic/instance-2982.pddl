(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot429 shot497 - shot
      ingredient339 ingredient365 ingredient38 ingredient81 - ingredient
      cocktail495 - cocktail
      dispenser80 dispenser152 dispenser353 dispenser102 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot429)
  (ontable shot497)
  (dispenses dispenser80 ingredient339)
  (dispenses dispenser152 ingredient365)
  (dispenses dispenser353 ingredient38)
  (dispenses dispenser102 ingredient81)
  (clean shaker219)
  (clean shot429)
  (clean shot497)
  (empty shaker219)
  (empty shot429)
  (empty shot497)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail495 ingredient38)
  (cocktail-part2 cocktail495 ingredient81)
)
 (:goal
  (and
      (contains shot429 cocktail495)
)))
