(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot338 shot456 shot476 - shot
      ingredient182 ingredient476 - ingredient
      cocktail51 - cocktail
      dispenser445 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot338)
  (ontable shot456)
  (ontable shot476)
  (dispenses dispenser445 ingredient182)
  (dispenses dispenser149 ingredient476)
  (clean shaker498)
  (clean shot338)
  (clean shot456)
  (clean shot476)
  (empty shaker498)
  (empty shot338)
  (empty shot456)
  (empty shot476)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail51 ingredient476)
  (cocktail-part2 cocktail51 ingredient182)
)
 (:goal
  (and
      (contains shot338 cocktail51)
      (contains shot456 ingredient182)
)))
