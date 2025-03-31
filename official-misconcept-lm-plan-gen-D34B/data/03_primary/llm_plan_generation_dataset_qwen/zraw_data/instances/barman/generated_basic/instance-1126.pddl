(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot498 shot288 - shot
      ingredient315 ingredient80 ingredient474 - ingredient
      cocktail1 - cocktail
      dispenser287 dispenser152 dispenser79 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot498)
  (ontable shot288)
  (dispenses dispenser287 ingredient315)
  (dispenses dispenser152 ingredient80)
  (dispenses dispenser79 ingredient474)
  (clean shaker221)
  (clean shot498)
  (clean shot288)
  (empty shaker221)
  (empty shot498)
  (empty shot288)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient80)
  (cocktail-part2 cocktail1 ingredient315)
)
 (:goal
  (and
      (contains shot498 cocktail1)
)))
