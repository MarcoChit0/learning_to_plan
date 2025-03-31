(define (problem prob)
 (:domain barman)
 (:objects 
      shaker23 - shaker
      left right - hand
      shot289 shot462 - shot
      ingredient12 ingredient456 ingredient194 ingredient421 - ingredient
      cocktail353 - cocktail
      dispenser362 dispenser169 dispenser353 dispenser484 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker23)
  (ontable shot289)
  (ontable shot462)
  (dispenses dispenser362 ingredient12)
  (dispenses dispenser169 ingredient456)
  (dispenses dispenser353 ingredient194)
  (dispenses dispenser484 ingredient421)
  (clean shaker23)
  (clean shot289)
  (clean shot462)
  (empty shaker23)
  (empty shot289)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker23 l0)
  (shaker-level shaker23 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail353 ingredient456)
  (cocktail-part2 cocktail353 ingredient12)
)
 (:goal
  (and
      (contains shot289 cocktail353)
)))
