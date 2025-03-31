(define (problem prob)
 (:domain barman)
 (:objects 
      shaker13 - shaker
      left right - hand
      shot340 shot487 - shot
      ingredient428 ingredient139 ingredient18 - ingredient
      cocktail1 - cocktail
      dispenser77 dispenser21 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker13)
  (ontable shot340)
  (ontable shot487)
  (dispenses dispenser77 ingredient428)
  (dispenses dispenser21 ingredient139)
  (dispenses dispenser303 ingredient18)
  (clean shaker13)
  (clean shot340)
  (clean shot487)
  (empty shaker13)
  (empty shot340)
  (empty shot487)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker13 l0)
  (shaker-level shaker13 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient18)
  (cocktail-part2 cocktail1 ingredient428)
)
 (:goal
  (and
      (contains shot340 cocktail1)
)))
