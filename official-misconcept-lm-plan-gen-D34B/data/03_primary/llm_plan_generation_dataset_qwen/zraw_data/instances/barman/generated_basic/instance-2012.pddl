(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot142 shot154 shot385 - shot
      ingredient338 ingredient68 ingredient184 - ingredient
      cocktail100 - cocktail
      dispenser7 dispenser131 dispenser58 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot142)
  (ontable shot154)
  (ontable shot385)
  (dispenses dispenser7 ingredient338)
  (dispenses dispenser131 ingredient68)
  (dispenses dispenser58 ingredient184)
  (clean shaker98)
  (clean shot142)
  (clean shot154)
  (clean shot385)
  (empty shaker98)
  (empty shot142)
  (empty shot154)
  (empty shot385)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail100 ingredient338)
  (cocktail-part2 cocktail100 ingredient68)
)
 (:goal
  (and
      (contains shot142 cocktail100)
      (contains shot154 ingredient68)
)))
