(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot268 shot477 shot14 - shot
      ingredient477 ingredient365 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser298 dispenser278 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot268)
  (ontable shot477)
  (ontable shot14)
  (dispenses dispenser298 ingredient477)
  (dispenses dispenser278 ingredient365)
  (clean shaker247)
  (clean shot268)
  (clean shot477)
  (clean shot14)
  (empty shaker247)
  (empty shot268)
  (empty shot477)
  (empty shot14)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient477)
  (cocktail-part2 cocktail1 ingredient365)
  (cocktail-part1 cocktail2 ingredient477)
  (cocktail-part2 cocktail2 ingredient365)
)
 (:goal
  (and
      (contains shot268 cocktail2)
      (contains shot477 cocktail1)
)))
