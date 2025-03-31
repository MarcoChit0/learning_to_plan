(define (problem prob)
 (:domain barman)
 (:objects 
      shaker223 - shaker
      left right - hand
      shot249 shot359 - shot
      ingredient338 ingredient119 - ingredient
      cocktail1 - cocktail
      dispenser283 dispenser313 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker223)
  (ontable shot249)
  (ontable shot359)
  (dispenses dispenser283 ingredient338)
  (dispenses dispenser313 ingredient119)
  (clean shaker223)
  (clean shot249)
  (clean shot359)
  (empty shaker223)
  (empty shot249)
  (empty shot359)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker223 l0)
  (shaker-level shaker223 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient338)
  (cocktail-part2 cocktail1 ingredient119)
)
 (:goal
  (and
      (contains shot249 cocktail1)
)))
