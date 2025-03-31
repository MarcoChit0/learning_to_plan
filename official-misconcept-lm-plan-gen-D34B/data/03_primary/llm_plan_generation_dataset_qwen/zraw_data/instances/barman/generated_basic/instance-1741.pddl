(define (problem prob)
 (:domain barman)
 (:objects 
      shaker115 - shaker
      left right - hand
      shot338 shot32 - shot
      ingredient229 ingredient324 ingredient133 - ingredient
      cocktail31 - cocktail
      dispenser97 dispenser321 dispenser65 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker115)
  (ontable shot338)
  (ontable shot32)
  (dispenses dispenser97 ingredient229)
  (dispenses dispenser321 ingredient324)
  (dispenses dispenser65 ingredient133)
  (clean shaker115)
  (clean shot338)
  (clean shot32)
  (empty shaker115)
  (empty shot338)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker115 l0)
  (shaker-level shaker115 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail31 ingredient324)
  (cocktail-part2 cocktail31 ingredient133)
)
 (:goal
  (and
      (contains shot338 cocktail31)
)))
