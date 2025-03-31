(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot392 - shot
      ingredient123 ingredient267 ingredient181 - ingredient
      cocktail1 - cocktail
      dispenser255 dispenser459 dispenser103 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot392)
  (dispenses dispenser255 ingredient123)
  (dispenses dispenser459 ingredient267)
  (dispenses dispenser103 ingredient181)
  (clean shaker170)
  (clean shot392)
  (empty shaker170)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient181)
  (cocktail-part2 cocktail1 ingredient267)
)
 (:goal
  (and
      (contains shot392 cocktail1)
)))
