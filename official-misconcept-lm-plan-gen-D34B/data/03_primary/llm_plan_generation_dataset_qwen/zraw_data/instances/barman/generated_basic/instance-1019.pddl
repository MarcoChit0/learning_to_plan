(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot250 - shot
      ingredient132 ingredient207 ingredient337 - ingredient
      cocktail1 - cocktail
      dispenser161 dispenser156 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot250)
  (dispenses dispenser161 ingredient132)
  (dispenses dispenser156 ingredient207)
  (dispenses dispenser292 ingredient337)
  (clean shaker157)
  (clean shot250)
  (empty shaker157)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient337)
  (cocktail-part2 cocktail1 ingredient132)
)
 (:goal
  (and
      (contains shot250 cocktail1)
)))
