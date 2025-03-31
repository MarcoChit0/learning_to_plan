(define (problem prob)
 (:domain barman)
 (:objects 
      shaker338 - shaker
      left right - hand
      shot122 shot259 shot117 - shot
      ingredient390 ingredient477 ingredient230 - ingredient
      cocktail1 - cocktail
      dispenser285 dispenser495 dispenser466 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker338)
  (ontable shot122)
  (ontable shot259)
  (ontable shot117)
  (dispenses dispenser285 ingredient390)
  (dispenses dispenser495 ingredient477)
  (dispenses dispenser466 ingredient230)
  (clean shaker338)
  (clean shot122)
  (clean shot259)
  (clean shot117)
  (empty shaker338)
  (empty shot122)
  (empty shot259)
  (empty shot117)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker338 l0)
  (shaker-level shaker338 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient230)
  (cocktail-part2 cocktail1 ingredient477)
)
 (:goal
  (and
      (contains shot122 cocktail1)
      (contains shot259 cocktail1)
)))
