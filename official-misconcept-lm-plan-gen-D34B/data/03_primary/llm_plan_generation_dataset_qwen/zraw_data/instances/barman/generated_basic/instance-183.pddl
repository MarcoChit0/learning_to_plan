(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot338 shot219 shot142 - shot
      ingredient435 ingredient181 ingredient135 ingredient339 - ingredient
      cocktail1 - cocktail
      dispenser76 dispenser218 dispenser104 dispenser255 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot338)
  (ontable shot219)
  (ontable shot142)
  (dispenses dispenser76 ingredient435)
  (dispenses dispenser218 ingredient181)
  (dispenses dispenser104 ingredient135)
  (dispenses dispenser255 ingredient339)
  (clean shaker114)
  (clean shot338)
  (clean shot219)
  (clean shot142)
  (empty shaker114)
  (empty shot338)
  (empty shot219)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient181)
  (cocktail-part2 cocktail1 ingredient435)
)
 (:goal
  (and
      (contains shot338 cocktail1)
      (contains shot219 cocktail1)
)))
