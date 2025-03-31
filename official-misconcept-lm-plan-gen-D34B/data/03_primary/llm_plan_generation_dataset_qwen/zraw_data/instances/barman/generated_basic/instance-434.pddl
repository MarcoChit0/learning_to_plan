(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot45 shot472 - shot
      ingredient380 ingredient141 ingredient170 - ingredient
      cocktail1 - cocktail
      dispenser157 dispenser236 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot45)
  (ontable shot472)
  (dispenses dispenser157 ingredient380)
  (dispenses dispenser236 ingredient141)
  (dispenses dispenser213 ingredient170)
  (clean shaker189)
  (clean shot45)
  (clean shot472)
  (empty shaker189)
  (empty shot45)
  (empty shot472)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient170)
  (cocktail-part2 cocktail1 ingredient380)
)
 (:goal
  (and
      (contains shot45 cocktail1)
)))
