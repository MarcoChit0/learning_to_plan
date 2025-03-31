(define (problem prob)
 (:domain barman)
 (:objects 
      shaker133 - shaker
      left right - hand
      shot348 shot265 shot204 - shot
      ingredient146 ingredient105 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser424 dispenser347 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker133)
  (ontable shot348)
  (ontable shot265)
  (ontable shot204)
  (dispenses dispenser424 ingredient146)
  (dispenses dispenser347 ingredient105)
  (clean shaker133)
  (clean shot348)
  (clean shot265)
  (clean shot204)
  (empty shaker133)
  (empty shot348)
  (empty shot265)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker133 l0)
  (shaker-level shaker133 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient105)
  (cocktail-part2 cocktail1 ingredient146)
  (cocktail-part1 cocktail2 ingredient146)
  (cocktail-part2 cocktail2 ingredient105)
)
 (:goal
  (and
      (contains shot348 cocktail2)
      (contains shot265 cocktail1)
)))
