(define (problem prob)
 (:domain barman)
 (:objects 
      shaker29 - shaker
      left right - hand
      shot149 - shot
      ingredient141 ingredient483 ingredient154 - ingredient
      cocktail1 - cocktail
      dispenser272 dispenser266 dispenser355 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker29)
  (ontable shot149)
  (dispenses dispenser272 ingredient141)
  (dispenses dispenser266 ingredient483)
  (dispenses dispenser355 ingredient154)
  (clean shaker29)
  (clean shot149)
  (empty shaker29)
  (empty shot149)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker29 l0)
  (shaker-level shaker29 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient141)
  (cocktail-part2 cocktail1 ingredient483)
)
 (:goal
  (and
      (contains shot149 cocktail1)
)))
