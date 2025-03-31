(define (problem prob)
 (:domain barman)
 (:objects 
      shaker313 - shaker
      left right - hand
      shot104 shot36 - shot
      ingredient452 ingredient390 ingredient85 ingredient420 - ingredient
      cocktail268 - cocktail
      dispenser6 dispenser76 dispenser112 dispenser215 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker313)
  (ontable shot104)
  (ontable shot36)
  (dispenses dispenser6 ingredient452)
  (dispenses dispenser76 ingredient390)
  (dispenses dispenser112 ingredient85)
  (dispenses dispenser215 ingredient420)
  (clean shaker313)
  (clean shot104)
  (clean shot36)
  (empty shaker313)
  (empty shot104)
  (empty shot36)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker313 l0)
  (shaker-level shaker313 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail268 ingredient452)
  (cocktail-part2 cocktail268 ingredient420)
)
 (:goal
  (and
      (contains shot104 cocktail268)
)))
